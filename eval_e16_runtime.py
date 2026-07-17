#!/usr/bin/env python3
"""
eval_e16_runtime.py — E16 Runtime / Deployment Benchmark
=========================================================
Measures inference latency, throughput, and memory footprint for each
SABRE+ agent and the full [B+K] triage pipeline.

Benchmark protocol
------------------
  Warm-up : 50 forward passes (discarded) to warm JIT/CUDA caches.
  Timing  : N_REPS=500 single-sample passes, N_BATCH_REPS=100 batch passes.
  Clock   : time.perf_counter() around the score_batch call (CPU-side wall time).
  GPU sync: torch.cuda.synchronize() before stopping the clock to flush CUDA queue.
  Memory  : peak RSS before/after loading all agents (psutil.Process.memory_info).
            GPU memory reported via torch.cuda.max_memory_allocated.

Metrics reported
----------------
  Single-sample: p50, p95, p99 latency (ms); throughput (samples/s)
  Batch sizes  : {1, 8, 32, 128, 256, 1024}  →  p50 latency + throughput
  MC-Dropout   : T=1 vs T=10 overhead ratio
  Model sizes  : parameter count + weight bytes
  Pipeline     : [S+B+K+MLP] end-to-end single-sample -- Semantic CNN
                 (T=10) -> Behavioural CNN (T=10) -> Knowledge CNN (T=10)
                 -> Decision MLP (single deterministic pass), genuinely
                 measured together. An earlier version of this script only
                 timed a [B+K]-mean-fusion pipeline that never called the
                 Semantic agent or the Decision MLP at all, which did not
                 match what the main paper's runtime table (tab:latency)
                 describes as "Full pipeline (3 agents + MLP)".

Outputs
-------
  results/e16_runtime.json
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

CKPT_DIR   = Path("checkpoints")
RESULTS    = Path("results")
SPLITS_DIR = Path("splits")
RANSKNOW   = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_718.csv"

N_REPS       = 500   # single-sample timing reps
N_BATCH_REPS = 100   # batch timing reps
WARMUP       = 50
T_MC         = 10
BATCH_SIZES  = [1, 8, 32, 128, 256, 1024]

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_Count", "Family_List",
    "Tool_List", "Dominant_Tactic", "Platform_Signal",
    # v4-only RansKnow metadata columns, not part of the 23-dim ATT&CK
    # feature set (see train_knowledge_agent.py's drop_cols for the same fix)
    "Year", "DurationSeconds", "Transcript_Provider",
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_single(fn, n_reps: int, warmup: int) -> dict:
    """Time fn() n_reps times after warmup. Returns percentile latencies (ms)."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(n_reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "p50_ms":  float(np.percentile(times, 50)),
        "p95_ms":  float(np.percentile(times, 95)),
        "p99_ms":  float(np.percentile(times, 99)),
        "mean_ms": float(times.mean()),
        "std_ms":  float(times.std()),
        "throughput_sps": float(1000.0 / np.percentile(times, 50)),
    }


def timed_batch(fn, batch_size: int, n_reps: int, warmup: int) -> dict:
    """Time fn() n_reps times. fn() processes batch_size samples."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(n_reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    p50 = float(np.percentile(times, 50))
    return {
        "batch_size":     batch_size,
        "p50_ms":         p50,
        "p95_ms":         float(np.percentile(times, 95)),
        "throughput_sps": float(batch_size * 1000.0 / p50),
    }


def model_size(model: torch.nn.Module) -> dict:
    params = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"params": params, "weight_kb": round(bytes_ / 1024, 1)}


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
# Knowledge lookup (non-neural, pure Python/NumPy)
# ---------------------------------------------------------------------------

def build_knowledge_lookup():
    """Corpus-mean-always: identical vector regardless of family name, same
    fix as eval_sabre2.py's build_knowledge_lookup(). A per-family lookup
    keyed by the sample's true family leaks that label; it also had a
    zero-vector fallback for families absent from RansKnow's Family_List
    (e.g. REvil), which this benchmark's timing loop used for a fixed
    "Conti" lookup anyway and so was not itself latency-relevant, but is
    removed here for consistency with the rest of the codebase."""
    import pandas as pd
    df_rk = pd.read_csv(RANSKNOW)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    corpus_mean = X_rk.mean(axis=0)
    n_feat = X_rk.shape[1]

    def lookup(cn: str) -> np.ndarray:
        return corpus_mean

    return lookup, n_feat


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"N_REPS : {N_REPS} single-sample, {N_BATCH_REPS} batch")

    results: dict = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_reps_single": N_REPS,
        "n_reps_batch":  N_BATCH_REPS,
        "warmup": WARMUP,
        "T_mc": T_MC,
    }

    # -----------------------------------------------------------------------
    # Synthetic inputs (no data loading overhead in the timing loop)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    x_beh_1   = rng.standard_normal((1,  len(FEATURE_COLS))).astype(np.float32)
    x_sem_1   = rng.standard_normal((1,  384)).astype(np.float32)
    # Knowledge feature dim determined at load time
    kn_lookup, kn_dim = build_knowledge_lookup()
    x_know_1  = rng.standard_normal((1,  kn_dim)).astype(np.float32)

    # -----------------------------------------------------------------------
    # A. Behavioral agent
    # -----------------------------------------------------------------------
    print("\n=== A. Behavioral Agent [B] ===")
    rss_before = rss_mb()
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    b_agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    rss_after = rss_mb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sz_b = model_size(b_agent.model)
    print(f"  Params: {sz_b['params']:,}  Weights: {sz_b['weight_kb']:.1f} KB  "
          f"RSS delta: {rss_after - rss_before:.1f} MB")

    # Single-sample T=10
    def beh_single():
        b_agent.score_batch([{"features": x_beh_1[0]}])

    lat_b = timed_single(beh_single, N_REPS, WARMUP)
    print(f"  Single (T={T_MC}): p50={lat_b['p50_ms']:.3f} ms  "
          f"p95={lat_b['p95_ms']:.3f} ms  p99={lat_b['p99_ms']:.3f} ms  "
          f"throughput={lat_b['throughput_sps']:.0f} sps")

    # Single T=1 (no MC-Dropout overhead)
    b_agent_t1 = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=1,
    )
    def beh_single_t1():
        b_agent_t1.score_batch([{"features": x_beh_1[0]}])

    lat_b_t1 = timed_single(beh_single_t1, N_REPS, WARMUP)
    overhead = lat_b["p50_ms"] / lat_b_t1["p50_ms"]
    print(f"  Single (T=1):  p50={lat_b_t1['p50_ms']:.3f} ms  "
          f"MC-Dropout T=10 overhead: {overhead:.2f}×")

    # Batch sweep
    print("  Batch throughput:")
    batch_b = []
    for bs in BATCH_SIZES:
        x_bs = rng.standard_normal((bs, len(FEATURE_COLS))).astype(np.float32)
        inputs = [{"features": x_bs[i]} for i in range(bs)]
        def beh_batch(inp=inputs):
            b_agent.score_batch(inp)
        res = timed_batch(beh_batch, bs, N_BATCH_REPS, WARMUP)
        batch_b.append(res)
        print(f"    bs={bs:>5}  p50={res['p50_ms']:>7.2f} ms  "
              f"throughput={res['throughput_sps']:>8.0f} sps")

    gpu_b = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    results["behavioral"] = {
        "model": sz_b,
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "gpu_peak_mb": round(gpu_b, 1),
        "single_T10": lat_b,
        "single_T1":  lat_b_t1,
        "mc_dropout_overhead_x": round(overhead, 2),
        "batch": batch_b,
    }

    # -----------------------------------------------------------------------
    # B. Knowledge agent
    # -----------------------------------------------------------------------
    print("\n=== B. Knowledge Agent [K] ===")
    rss_before = rss_mb()
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    k_agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    rss_after = rss_mb()

    sz_k = model_size(k_agent.model)
    print(f"  Params: {sz_k['params']:,}  Weights: {sz_k['weight_kb']:.1f} KB  "
          f"RSS delta: {rss_after - rss_before:.1f} MB")

    def know_single():
        k_agent.score_batch([{"features": x_know_1[0]}])

    lat_k = timed_single(know_single, N_REPS, WARMUP)
    print(f"  Single (T={T_MC}): p50={lat_k['p50_ms']:.3f} ms  "
          f"p95={lat_k['p95_ms']:.3f} ms  p99={lat_k['p99_ms']:.3f} ms  "
          f"throughput={lat_k['throughput_sps']:.0f} sps")

    # Batch sweep
    print("  Batch throughput:")
    batch_k = []
    for bs in BATCH_SIZES:
        x_bs = rng.standard_normal((bs, kn_dim)).astype(np.float32)
        inputs = [{"features": x_bs[i]} for i in range(bs)]
        def know_batch(inp=inputs):
            k_agent.score_batch(inp)
        res = timed_batch(know_batch, bs, N_BATCH_REPS, WARMUP)
        batch_k.append(res)
        print(f"    bs={bs:>5}  p50={res['p50_ms']:>7.2f} ms  "
              f"throughput={res['throughput_sps']:>8.0f} sps")

    results["knowledge"] = {
        "model": sz_k,
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "single_T10": lat_k,
        "batch": batch_k,
    }

    # -----------------------------------------------------------------------
    # C. Knowledge lookup (non-neural dictionary step)
    # -----------------------------------------------------------------------
    print("\n=== C. Knowledge Lookup (dict + numpy, no neural network) ===")
    family_name = "Conti"

    def know_lookup():
        vec = kn_lookup(family_name)
        return vec

    lat_kl = timed_single(know_lookup, N_REPS, warmup=10)
    print(f"  Lookup only: p50={lat_kl['p50_ms']*1000:.3f} µs  "
          f"(p95={lat_kl['p95_ms']*1000:.3f} µs)")

    results["knowledge_lookup"] = {
        "p50_us":  round(lat_kl["p50_ms"] * 1000, 3),
        "p95_us":  round(lat_kl["p95_ms"] * 1000, 3),
        "p99_us":  round(lat_kl["p99_ms"] * 1000, 3),
    }

    # -----------------------------------------------------------------------
    # D. Semantic agent (CNN only — embeddings pre-computed offline)
    # -----------------------------------------------------------------------
    print("\n=== D. Semantic Agent [S] (CNN on pre-computed 384-dim embedding) ===")
    rss_before = rss_mb()
    from Agentic_SABRE.agents.semantic import SemanticAgent
    s_agent = SemanticAgent(T=T_MC)
    rss_after = rss_mb()

    sz_s = model_size(s_agent.model)
    print(f"  Params: {sz_s['params']:,}  Weights: {sz_s['weight_kb']:.1f} KB  "
          f"RSS delta: {rss_after - rss_before:.1f} MB")
    print(f"  Note: embedding (all-MiniLM-L6-v2, 22.7M params) is pre-computed offline.")

    def sem_single():
        s_agent.score_batch([{"embedding": x_sem_1[0]}])

    lat_s = timed_single(sem_single, N_REPS, WARMUP)
    print(f"  Single (T={T_MC}): p50={lat_s['p50_ms']:.3f} ms  "
          f"p95={lat_s['p95_ms']:.3f} ms  p99={lat_s['p99_ms']:.3f} ms  "
          f"throughput={lat_s['throughput_sps']:.0f} sps")

    batch_s = []
    print("  Batch throughput:")
    for bs in BATCH_SIZES:
        x_bs = rng.standard_normal((bs, 384)).astype(np.float32)
        inputs = [{"embedding": x_bs[i]} for i in range(bs)]
        def sem_batch(inp=inputs):
            s_agent.score_batch(inp)
        res = timed_batch(sem_batch, bs, N_BATCH_REPS, WARMUP)
        batch_s.append(res)
        print(f"    bs={bs:>5}  p50={res['p50_ms']:>7.2f} ms  "
              f"throughput={res['throughput_sps']:>8.0f} sps")

    results["semantic"] = {
        "model": sz_s,
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "single_T10": lat_s,
        "batch": batch_s,
        "note": "CNN only; embedding (all-MiniLM-L6-v2, 22.7M params) pre-computed offline",
    }

    # -----------------------------------------------------------------------
    # E1. Decision MLP alone (single deterministic pass, no MC Dropout)
    # -----------------------------------------------------------------------
    print("\n=== E1. Decision MLP [S+B+K -> MLP] (single deterministic pass) ===")
    from Agentic_SABRE.agents.decision import DecisionAgent
    d_agent = DecisionAgent(
        artifact_path=str(CKPT_DIR / "decision_fuser_3agent.json"),
        use_uncertainties=True,
    )
    mlp_input = {"p_sem": 0.5, "p_beh": 0.5, "p_know": 0.5,
                 "σ_sem": 0.01, "σ_beh": 0.05, "σ_know": 0.01}

    def mlp_single():
        d_agent.score(mlp_input)

    lat_mlp = timed_single(mlp_single, N_REPS, WARMUP)
    print(f"  Single pass: p50={lat_mlp['p50_ms']:.3f} ms  p95={lat_mlp['p95_ms']:.3f} ms  "
          f"throughput={lat_mlp['throughput_sps']:.0f} sps")
    results["decision_mlp"] = {"single_pass": lat_mlp}

    # -----------------------------------------------------------------------
    # E2. Full [S+B+K+MLP] pipeline (single-sample, end-to-end, genuinely
    #     measured together -- Semantic + Behavioural + Knowledge (each
    #     T=10 MC Dropout) followed by the Decision MLP (single pass))
    # -----------------------------------------------------------------------
    print("\n=== E2. Full [S+B+K+MLP] pipeline (end-to-end single sample) ===")
    print("  Sequence: S-agent CNN (T=10) -> B-agent CNN (T=10) -> "
          "K-agent CNN (T=10) -> Decision MLP (single pass)")

    x_beh_s  = x_beh_1[0]
    x_sem_s  = x_sem_1[0]

    def pipeline_full():
        kv = kn_lookup("Conti")
        s_out = s_agent.score_batch([{"embedding": x_sem_s}])[0]
        b_out = b_agent.score_batch([{"features": x_beh_s}])[0]
        k_out = k_agent.score_batch([{"features": kv}])[0]
        fused = d_agent.score({
            "p_sem": s_out.prob, "p_beh": b_out.prob, "p_know": k_out.prob,
            "σ_sem": s_out.uncert, "σ_beh": b_out.uncert, "σ_know": k_out.uncert,
        })
        return fused.prob

    lat_full = timed_single(pipeline_full, N_REPS, WARMUP)
    print(f"  [S+B+K+MLP] pipeline: p50={lat_full['p50_ms']:.3f} ms  "
          f"p95={lat_full['p95_ms']:.3f} ms  p99={lat_full['p99_ms']:.3f} ms  "
          f"throughput={lat_full['throughput_sps']:.0f} sps")

    # Agent-only reference (same sequence, no Decision MLP call at the end)
    def pipeline_agents_only():
        kv = kn_lookup("Conti")
        s_agent.score_batch([{"embedding": x_sem_s}])
        b_agent.score_batch([{"features": x_beh_s}])
        k_agent.score_batch([{"features": kv}])

    lat_agents_only = timed_single(pipeline_agents_only, N_REPS, WARMUP)
    print(f"  Agent-only (no MLP): p50={lat_agents_only['p50_ms']:.3f} ms")
    delta_mlp_ms = lat_full["p50_ms"] - lat_agents_only["p50_ms"]
    print(f"  Delta vs agent-only (no MLP): +{delta_mlp_ms:.3f} ms")

    # Batch pipeline sweep
    print("  Batch [S+B+K+MLP] throughput:")
    batch_full = []
    for bs in BATCH_SIZES:
        x_beh_bs = rng.standard_normal((bs, len(FEATURE_COLS))).astype(np.float32)
        x_sem_bs = rng.standard_normal((bs, 384)).astype(np.float32)
        x_know_bs = np.stack([kn_lookup("Conti")] * bs)
        s_inp = [{"embedding": x_sem_bs[i]} for i in range(bs)]
        b_inp = [{"features": x_beh_bs[i]}  for i in range(bs)]
        k_inp = [{"features": x_know_bs[i]} for i in range(bs)]
        def pipe_batch(si=s_inp, bi=b_inp, ki=k_inp, bsz=bs):
            s_outs = s_agent.score_batch(si)
            b_outs = b_agent.score_batch(bi)
            k_outs = k_agent.score_batch(ki)
            _ = [d_agent.score({
                    "p_sem": s.prob, "p_beh": b.prob, "p_know": k.prob,
                    "σ_sem": s.uncert, "σ_beh": b.uncert, "σ_know": k.uncert,
                 }) for s, b, k in zip(s_outs, b_outs, k_outs)]
        res = timed_batch(pipe_batch, bs, N_BATCH_REPS, WARMUP)
        batch_full.append(res)
        print(f"    bs={bs:>5}  p50={res['p50_ms']:>7.2f} ms  "
              f"throughput={res['throughput_sps']:>8.0f} sps")

    results["pipeline_full"] = {
        "single_T10": lat_full,
        "agents_only_no_mlp": lat_agents_only,
        "delta_mlp_ms": round(delta_mlp_ms, 4),
        "batch": batch_full,
        "description": ("S-CNN(T=10) + B-CNN(T=10) + K-CNN(T=10) + Decision MLP (single pass), "
                         "genuinely measured together -- this is what the main paper's "
                         "'Full pipeline (3 agents + MLP)' row reports"),
    }
    # Kept for backward compatibility with any downstream consumer of the old key.
    lat_bk = lat_full

    # -----------------------------------------------------------------------
    # F. Triage queue simulation: sustained throughput over 10-second window
    # -----------------------------------------------------------------------
    print("\n=== F. Triage queue context ===")
    window_s = 10.0    # telemetry window duration (seconds)
    p50_full = lat_full["p50_ms"] / 1000.0
    latency_budget_pct = (p50_full / window_s) * 100

    print(f"  10-second window duration:   {window_s:.0f} s")
    print(f"  [S+B+K+MLP] p50 latency:     {lat_full['p50_ms']:.3f} ms")
    print(f"  Latency / window budget:     {latency_budget_pct:.4f}%")
    print(f"  Concurrent streams at p50:   {int(window_s / p50_full):,} streams")
    print(f"  [B]          p50 throughput: {lat_b['throughput_sps']:.0f} sps")
    print(f"  [S+B+K+MLP]  p50 throughput: {lat_full['throughput_sps']:.0f} sps")

    results["triage_queue"] = {
        "window_duration_s":       window_s,
        "full_pipeline_p50_ms":    lat_full["p50_ms"],
        "latency_budget_pct":      round(latency_budget_pct, 4),
        "concurrent_streams_p50":  int(window_s / p50_full),
        "b_throughput_sps":        lat_b["throughput_sps"],
        "full_pipeline_throughput_sps": lat_full["throughput_sps"],
    }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  E16 RUNTIME BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Component':<18} {'Params':>10} {'Wts(KB)':>8} {'p50(ms)':>9} "
          f"{'p95(ms)':>9} {'throughput':>12}")
    print("  " + "-" * 69)

    mlp_params = sum(p.numel() for p in d_agent.trainer.model.parameters())
    rows = [
        ("Semantic [S]",    sz_s["params"], sz_s["weight_kb"], lat_s["p50_ms"],   lat_s["p95_ms"],   lat_s["throughput_sps"]),
        ("Behavioural [B]", sz_b["params"], sz_b["weight_kb"], lat_b["p50_ms"],   lat_b["p95_ms"],   lat_b["throughput_sps"]),
        ("Knowledge [K]",   sz_k["params"], sz_k["weight_kb"], lat_k["p50_ms"],   lat_k["p95_ms"],   lat_k["throughput_sps"]),
        ("Decision MLP",    mlp_params,     None,               lat_mlp["p50_ms"], lat_mlp["p95_ms"], lat_mlp["throughput_sps"]),
        ("Full [S+B+K+MLP]", sz_s["params"]+sz_b["params"]+sz_k["params"]+mlp_params, None,
                             lat_full["p50_ms"], lat_full["p95_ms"], lat_full["throughput_sps"]),
    ]
    for name, params, wt, p50, p95, tp in rows:
        wt_s = f"{wt:>8.1f}" if wt is not None else f"{'--':>8}"
        print(f"  {name:<18} {params:>10,} {wt_s} {p50:>9.3f} {p95:>9.3f} {tp:>11.0f} sps")

    print(f"\n  MC-Dropout overhead (T=10 vs T=1, [B]): {overhead:.2f}×")
    print(f"  Delta vs agent-only (no MLP):            +{delta_mlp_ms:.3f} ms")
    print(f"  Knowledge lookup (dict):                "
          f"{lat_kl['p50_ms']*1000:.3f} µs (negligible)")
    print(f"  Concurrent streams supported (p50):     "
          f"{int(window_s / p50_full):,} at [S+B+K+MLP] single-sample latency")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = RESULTS / "e16_runtime.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
