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
  Pipeline     : [B+K] end-to-end single-sample (incl. knowledge lookup)

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
RANSKNOW   = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"

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
    import pandas as pd
    df_rk = pd.read_csv(RANSKNOW)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    family_feats: dict = {}
    for i, row in df_rk.iterrows():
        for fam in [f.strip() for f in str(row.get("Family_List", "")).split(",")]:
            if fam and fam != "nan":
                family_feats.setdefault(fam, []).append(X_rk[i])
    family_mean = {f: np.mean(np.stack(v), axis=0) for f, v in family_feats.items()}
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)
    n_feat = X_rk.shape[1]

    def lookup(cn: str) -> np.ndarray:
        if cn in family_mean:
            return family_mean[cn]
        for k in family_mean:
            if k.lower() in cn.lower() or cn.lower() in k.lower():
                return family_mean[k]
        return zero_vec

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
    # E. Full [B+K] pipeline (single-sample, end-to-end)
    # -----------------------------------------------------------------------
    print("\n=== E. Full [B+K] pipeline (end-to-end single sample) ===")
    print("  Sequence: knowledge lookup → K-agent CNN → B-agent CNN → mean fusion")

    x_beh_s  = x_beh_1[0]
    x_know_s = x_know_1[0]

    def pipeline_bk():
        # 1. Knowledge lookup (O(1) dict)
        kv = kn_lookup("Conti")
        # 2. Knowledge CNN
        k_out = k_agent.score_batch([{"features": kv}])[0]
        # 3. Behavioral CNN
        b_out = b_agent.score_batch([{"features": x_beh_s}])[0]
        # 4. Mean fusion + triage decision
        fused = (b_out.prob + k_out.prob) / 2.0
        uncert = b_out.uncert
        if fused >= 0.5:
            decision = "DETECT"
        elif uncert >= 0.10:
            decision = "ESCALATE"
        else:
            decision = "BENIGN"
        return decision

    lat_bk = timed_single(pipeline_bk, N_REPS, WARMUP)
    print(f"  [B+K] pipeline: p50={lat_bk['p50_ms']:.3f} ms  "
          f"p95={lat_bk['p95_ms']:.3f} ms  p99={lat_bk['p99_ms']:.3f} ms  "
          f"throughput={lat_bk['throughput_sps']:.0f} sps")

    # Batch pipeline sweep
    print("  Batch [B+K] throughput:")
    batch_bk = []
    for bs in BATCH_SIZES:
        x_beh_bs  = rng.standard_normal((bs, len(FEATURE_COLS))).astype(np.float32)
        x_know_bs = np.stack([kn_lookup("Conti")] * bs)
        b_inp  = [{"features": x_beh_bs[i]}  for i in range(bs)]
        k_inp  = [{"features": x_know_bs[i]} for i in range(bs)]
        def pipe_batch(bi=b_inp, ki=k_inp, bsz=bs):
            b_outs = b_agent.score_batch(bi)
            k_outs = k_agent.score_batch(ki)
            _ = [(b.prob + k.prob) / 2 for b, k in zip(b_outs, k_outs)]
        res = timed_batch(pipe_batch, bs, N_BATCH_REPS, WARMUP)
        batch_bk.append(res)
        print(f"    bs={bs:>5}  p50={res['p50_ms']:>7.2f} ms  "
              f"throughput={res['throughput_sps']:>8.0f} sps")

    results["pipeline_bk"] = {
        "single_T10": lat_bk,
        "batch": batch_bk,
        "description": "knowledge_lookup + K-CNN (T=10) + B-CNN (T=10) + mean fusion + triage",
    }

    # -----------------------------------------------------------------------
    # F. Triage queue simulation: sustained throughput over 10-second window
    # -----------------------------------------------------------------------
    print("\n=== F. Triage queue context ===")
    window_s = 10.0    # telemetry window duration (seconds)
    p50_bk   = lat_bk["p50_ms"] / 1000.0
    windows_per_second = 1.0 / p50_bk
    latency_budget_pct = (p50_bk / window_s) * 100

    print(f"  10-second window duration: {window_s:.0f} s")
    print(f"  [B+K] p50 latency:         {lat_bk['p50_ms']:.3f} ms")
    print(f"  Latency / window budget:   {latency_budget_pct:.4f}%")
    print(f"  Concurrent streams at p50: {int(window_s / p50_bk):,} streams")
    print(f"  [B]  p50 throughput:       {lat_b['throughput_sps']:.0f} sps")
    print(f"  [B+K] p50 throughput:      {lat_bk['throughput_sps']:.0f} sps")

    results["triage_queue"] = {
        "window_duration_s":       window_s,
        "bk_p50_ms":               lat_bk["p50_ms"],
        "latency_budget_pct":      round(latency_budget_pct, 4),
        "concurrent_streams_p50":  int(window_s / p50_bk),
        "b_throughput_sps":        lat_b["throughput_sps"],
        "bk_throughput_sps":       lat_bk["throughput_sps"],
    }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  E16 RUNTIME BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Agent':<12} {'Params':>10} {'Wts(KB)':>8} {'p50(ms)':>9} "
          f"{'p95(ms)':>9} {'throughput':>12}")
    print("  " + "-" * 63)

    rows = [
        ("[B]",    sz_b["params"], sz_b["weight_kb"],  lat_b["p50_ms"],  lat_b["p95_ms"],  lat_b["throughput_sps"]),
        ("[K]",    sz_k["params"], sz_k["weight_kb"],  lat_k["p50_ms"],  lat_k["p95_ms"],  lat_k["throughput_sps"]),
        ("[S]",    sz_s["params"], sz_s["weight_kb"],  lat_s["p50_ms"],  lat_s["p95_ms"],  lat_s["throughput_sps"]),
        ("[B+K]",  sz_b["params"]+sz_k["params"], sz_b["weight_kb"]+sz_k["weight_kb"],
                   lat_bk["p50_ms"], lat_bk["p95_ms"], lat_bk["throughput_sps"]),
    ]
    for name, params, wt, p50, p95, tp in rows:
        print(f"  {name:<12} {params:>10,} {wt:>8.1f} {p50:>9.3f} {p95:>9.3f} {tp:>11.0f} sps")

    print(f"\n  MC-Dropout overhead (T=10 vs T=1, [B]): {overhead:.2f}×")
    print(f"  Knowledge lookup (dict):                "
          f"{lat_kl['p50_ms']*1000:.3f} µs (negligible)")
    print(f"  Concurrent streams supported (p50):     "
          f"{int(window_s / p50_bk):,} at [B+K] single-sample latency")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = RESULTS / "e16_runtime.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
