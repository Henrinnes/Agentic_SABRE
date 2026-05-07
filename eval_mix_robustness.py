#!/usr/bin/env python3
"""
eval_mix_robustness.py — Mix Partition Robustness Evaluation
=============================================================
Tests whether the behavioral (and [B+K]) agent can detect ransomware
when a benign application runs CONCURRENTLY in the same trace.

Mix partition: 1,944 windows, 10 class combinations (benign×ransomware),
all label=1.  Hardware: mix/i3-gen12/ddr4-2133-16g — never seen during training.

Evaluation
----------
  [B]   : BehavioralAgent (random_stratified checkpoint, T=10 MC-Dropout)
  [B+K] : [B] fused with KnowledgeAgent via ransomware-family lookup in RansKnow

Baseline comparison
-------------------
  Load solo (non-concurrent) Conti and REvil windows from the
  random_stratified TEST split and compute detection rates.
  Delta = concurrent_detection - solo_detection per family.

Outputs
-------
  results/mix_robustness.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

MIX_PARQUET      = "/home/henry_k/Datasets/ransmap_extracted/ransmap_mix.parquet"
RANSMAP_PARQUET  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RANSKNOW_CSV     = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"
SPLITS_DIR       = Path("splits")
CKPT_DIR         = Path("checkpoints")
RESULTS_DIR      = Path("results")

SEED  = 42
T_MC  = 10    # MC-Dropout passes for uncertainty
N_BOOT = 1000  # bootstrap resamples for 95% CI on detection rate

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

# Ransomware families present in RansKnow
RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"}


def extract_ransom_family(class_name: str) -> str:
    """
    Parse ransomware family from a concurrent class name.
    e.g. 'AESCrypt_Conti' → 'Conti', 'Firefox_Office_REvil' → 'REvil'
    """
    for fam in RANSOMWARE_FAMILIES:
        if fam in class_name:
            return fam
    return ""


def build_knowledge_lookup() -> tuple[dict, np.ndarray, callable, list]:
    df_rk = pd.read_csv(RANSKNOW_CSV)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)

    family_feats: dict[str, list] = {}
    for i, row in df_rk.iterrows():
        flist = str(row.get("Family_List", ""))
        for fam in [f.strip() for f in flist.split(",")]:
            if fam and fam != "nan":
                family_feats.setdefault(fam, []).append(X_rk[i])

    family_mean = {fam: np.mean(np.stack(vecs), axis=0)
                   for fam, vecs in family_feats.items()}
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)

    def lookup(family: str) -> np.ndarray:
        if family in family_mean:
            return family_mean[family]
        for k in family_mean:
            if k.lower() in family.lower() or family.lower() in k.lower():
                return family_mean[k]
        return zero_vec

    return family_mean, zero_vec, lookup, feat_cols


def score_behavioral(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, uncertainties) via MC-Dropout T=10."""
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X[i]} for i in range(len(X))])
    probs  = np.array([o.prob   for o in outs])
    uncert = np.array([o.uncert for o in outs])
    return probs, uncert


def score_knowledge(X_know: np.ndarray) -> np.ndarray:
    """Returns probs from KnowledgeAgent on pre-built feature vectors."""
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X_know[i]} for i in range(len(X_know))])
    return np.array([o.prob for o in outs])


def bootstrap_detection_rate(detected: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% CI on detection rate (mean of detected==1 array)."""
    rng = np.random.default_rng(seed)
    rates = [rng.choice(detected, size=len(detected), replace=True).mean()
             for _ in range(n)]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def load_solo_baseline(families: list[str]) -> dict[str, dict]:
    """
    Load detection stats for solo (non-concurrent) ransomware windows from
    the random_stratified TEST split. Returns per-family dict with
    {detection_rate, mean_prob, mean_uncert, n_windows}.
    """
    sess_file = SPLITS_DIR / "random_stratified" / "test_sessions.txt"
    sessions  = set(sess_file.read_text().splitlines())
    df = pd.read_parquet(RANSMAP_PARQUET)
    df_test = df[df["session"].isin(sessions)].reset_index(drop=True)
    # Solo ransomware only (label=1)
    df_ransom = df_test[df_test["label"] == 1].reset_index(drop=True)

    results: dict[str, dict] = {}
    for fam in families:
        sub = df_ransom[df_ransom["class_name"].str.contains(fam, case=False, na=False)]
        if len(sub) == 0:
            continue
        X = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
        probs, uncerts = score_behavioral(X)
        detected = (probs >= 0.5).astype(int)
        lo, hi = bootstrap_detection_rate(detected)
        results[fam] = {
            "n_windows":      int(len(sub)),
            "detection_rate": float(detected.mean()),
            "ci95":           [lo, hi],
            "mean_prob":      float(probs.mean()),
            "mean_uncert":    float(uncerts.mean()),
        }
        print(f"  Solo {fam:12s}: n={len(sub):4d}  "
              f"DR={detected.mean():.3f} [{lo:.3f},{hi:.3f}]  "
              f"prob={probs.mean():.3f}  uncert={uncerts.mean():.3f}")
    return results


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Load mix partition
    # ------------------------------------------------------------------
    print("Loading mix partition ...")
    df_mix = pd.read_parquet(MIX_PARQUET)
    print(f"  {len(df_mix):,} windows  |  all label=1  |  "
          f"{df_mix['class_name'].nunique()} class combos")

    X_mix    = df_mix[FEATURE_COLS].to_numpy(dtype=np.float32)
    classes  = df_mix["class_name"].values
    sessions = df_mix["session"].values

    # Parse ransomware family from each class_name
    ransom_families = np.array([extract_ransom_family(c) for c in classes])
    print(f"  Ransomware components: {np.unique(ransom_families).tolist()}")

    # ------------------------------------------------------------------
    # Build knowledge lookup
    # ------------------------------------------------------------------
    print("\nBuilding knowledge lookup ...")
    family_mean, zero_vec, kn_lookup, kn_feat_cols = build_knowledge_lookup()
    X_know = np.stack([kn_lookup(fam) for fam in ransom_families])
    print(f"  Knowledge feature dim: {X_know.shape[1]}")
    # Flag windows where knowledge lookup succeeded (non-zero vector)
    know_hit = (X_know != zero_vec).any(axis=1)
    print(f"  Knowledge hit rate: {know_hit.mean():.1%} ({know_hit.sum()}/{len(know_hit)})")

    # ------------------------------------------------------------------
    # [B] Behavioral scoring
    # ------------------------------------------------------------------
    print("\n[B] Scoring behavioral agent ...")
    b_probs, b_uncerts = score_behavioral(X_mix)
    b_detected = (b_probs >= 0.5).astype(int)
    print(f"  Overall detection rate: {b_detected.mean():.4f}  "
          f"mean_prob={b_probs.mean():.4f}  mean_uncert={b_uncerts.mean():.4f}")

    # ------------------------------------------------------------------
    # [K] Knowledge scoring
    # ------------------------------------------------------------------
    print("\n[K] Scoring knowledge agent ...")
    k_probs = score_knowledge(X_know)

    # ------------------------------------------------------------------
    # [B+K] Fusion
    # ------------------------------------------------------------------
    bk_probs    = (b_probs + k_probs) / 2.0
    bk_detected = (bk_probs >= 0.5).astype(int)
    print(f"\n[B+K] Overall detection rate: {bk_detected.mean():.4f}  "
          f"mean_prob={bk_probs.mean():.4f}")

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------
    print("\n--- Per class_name breakdown ---")
    print(f"  {'Class':<28} {'N':>5}  {'DR_B':>6}  {'CI95_B':^15}  "
          f"{'DR_BK':>6}  {'MProb_B':>8}  {'MUncert_B':>9}")
    print("  " + "-" * 88)

    per_class = {}
    class_names = sorted(df_mix["class_name"].unique())
    for cn in class_names:
        mask = classes == cn
        n    = int(mask.sum())
        pb   = b_probs[mask]
        pb_d = b_detected[mask]
        pk   = k_probs[mask]
        pb_u = b_uncerts[mask]
        pbk  = bk_probs[mask]
        pbk_d = bk_detected[mask]

        dr_b  = float(pb_d.mean())
        dr_bk = float(pbk_d.mean())
        lo, hi = bootstrap_detection_rate(pb_d)

        print(f"  {cn:<28} {n:>5}  {dr_b:>6.3f}  [{lo:.3f},{hi:.3f}]  "
              f"{dr_bk:>6.3f}  {pb.mean():>8.3f}  {pb_u.mean():>9.3f}")

        per_class[cn] = {
            "n_windows":         n,
            "ransom_family":     extract_ransom_family(cn),
            "detection_rate_B":  dr_b,
            "ci95_B":            [lo, hi],
            "detection_rate_BK": dr_bk,
            "mean_prob_B":       float(pb.mean()),
            "mean_prob_K":       float(pk.mean()),
            "mean_prob_BK":      float(pbk.mean()),
            "mean_uncert_B":     float(pb_u.mean()),
        }

    # ------------------------------------------------------------------
    # Per-ransomware-family breakdown (aggregate across benign co-runners)
    # ------------------------------------------------------------------
    print("\n--- Per ransomware family (aggregated across co-running benign) ---")
    print(f"  {'RansomFam':<12} {'N':>5}  {'DR_B':>6}  {'CI95_B':^15}  {'DR_BK':>6}  {'MUncert':>8}")
    print("  " + "-" * 62)

    per_family_concurrent = {}
    for fam in sorted(np.unique(ransom_families)):
        if not fam:
            continue
        mask  = ransom_families == fam
        n     = int(mask.sum())
        pb    = b_probs[mask]
        pb_d  = b_detected[mask]
        pbk_d = bk_detected[mask]
        pb_u  = b_uncerts[mask]
        lo, hi = bootstrap_detection_rate(pb_d)

        print(f"  {fam:<12} {n:>5}  {pb_d.mean():>6.3f}  [{lo:.3f},{hi:.3f}]  "
              f"{pbk_d.mean():>6.3f}  {pb_u.mean():>8.3f}")

        per_family_concurrent[fam] = {
            "n_windows":         n,
            "detection_rate_B":  float(pb_d.mean()),
            "ci95_B":            [lo, hi],
            "detection_rate_BK": float(pbk_d.mean()),
            "mean_prob_B":       float(pb.mean()),
            "mean_uncert_B":     float(pb_u.mean()),
        }

    # ------------------------------------------------------------------
    # Solo baseline (non-concurrent ransomware from test split)
    # ------------------------------------------------------------------
    print("\n--- Solo baseline from random_stratified test split ---")
    solo_families = list({fam for fam in ransom_families if fam})
    solo_stats = load_solo_baseline(solo_families)

    # ------------------------------------------------------------------
    # Delta: concurrent vs. solo detection rate
    # ------------------------------------------------------------------
    print("\n--- Concurrent vs. Solo detection rate delta ---")
    print(f"  {'Family':<12} {'Solo_DR':>8} {'Conc_DR':>8} {'Delta':>8}")
    print("  " + "-" * 42)
    deltas = {}
    for fam in sorted(solo_stats.keys()):
        if fam not in per_family_concurrent:
            continue
        solo_dr = solo_stats[fam]["detection_rate"]
        conc_dr = per_family_concurrent[fam]["detection_rate_B"]
        delta   = conc_dr - solo_dr
        print(f"  {fam:<12} {solo_dr:>8.3f} {conc_dr:>8.3f} {delta:>+8.3f}")
        deltas[fam] = {"solo": solo_dr, "concurrent": conc_dr, "delta": delta}

    # ------------------------------------------------------------------
    # Uncertainty: is concurrent execution less certain?
    # ------------------------------------------------------------------
    print("\n--- Uncertainty distribution (B) ---")
    uncert_pcts = np.percentile(b_uncerts, [25, 50, 75, 90, 95])
    print(f"  p25={uncert_pcts[0]:.4f}  p50={uncert_pcts[1]:.4f}  "
          f"p75={uncert_pcts[2]:.4f}  p90={uncert_pcts[3]:.4f}  p95={uncert_pcts[4]:.4f}")
    high_uncert = (b_uncerts >= 0.10).mean()
    print(f"  High-uncertainty (≥0.10) windows: {high_uncert:.1%}")

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  MIX PARTITION ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"  N windows      : {len(df_mix):,}  (all ransomware, concurrent execution)")
    print(f"  N class combos : {len(class_names)}")
    print(f"  Hardware       : mix/i3-gen12/ddr4-2133-16g (unseen partition)")

    lo_b, hi_b = bootstrap_detection_rate(b_detected)
    lo_bk, hi_bk = bootstrap_detection_rate(bk_detected)
    print(f"\n  [B]   Detection rate: {b_detected.mean():.4f} [{lo_b:.4f},{hi_b:.4f}]  "
          f"mean_prob={b_probs.mean():.4f}  mean_uncert={b_uncerts.mean():.4f}")
    print(f"  [B+K] Detection rate: {bk_detected.mean():.4f} [{lo_bk:.4f},{hi_bk:.4f}]  "
          f"mean_prob={bk_probs.mean():.4f}")

    # AUPRC/AUROC not meaningful here since all label=1; report detection rate as key metric.
    # But we can compute AUPRC on a balanced sample (1944 ransom + same-size benign from test).
    print("\n  (AUPRC/AUROC require benign samples — see combined eval below)")

    # Compute combined AUPRC: mix windows as positive, test benign as negative
    sess_file = SPLITS_DIR / "random_stratified" / "test_sessions.txt"
    test_sess = set(sess_file.read_text().splitlines())
    df_all    = pd.read_parquet(RANSMAP_PARQUET)
    df_benign = df_all[(df_all["session"].isin(test_sess)) & (df_all["label"] == 0)].reset_index(drop=True)
    X_ben  = df_benign[FEATURE_COLS].to_numpy(dtype=np.float32)
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    print("\n  Scoring benign test windows for combined AUPRC ...")
    b_agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    ben_outs  = b_agent.score_batch([{"features": X_ben[i]} for i in range(len(X_ben))])
    ben_probs = np.array([o.prob for o in ben_outs])

    combined_probs = np.concatenate([b_probs, ben_probs])
    combined_y     = np.concatenate([np.ones(len(b_probs)), np.zeros(len(ben_probs))])
    comb_auprc = float(average_precision_score(combined_y, combined_probs))
    comb_auroc = float(roc_auc_score(combined_y, combined_probs))
    print(f"  Combined [B]  AUPRC={comb_auprc:.4f}  AUROC={comb_auroc:.4f}  "
          f"(mix-ransom vs test-benign)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = {
        "dataset": "ransmap_mix",
        "n_windows": int(len(df_mix)),
        "n_class_combos": int(len(class_names)),
        "hardware": "mix/i3-gen12/ddr4-2133-16g",
        "checkpoint": "behavioral_cnn_random_stratified.pt",
        "overall": {
            "B": {
                "detection_rate": float(b_detected.mean()),
                "ci95":           [lo_b, hi_b],
                "mean_prob":      float(b_probs.mean()),
                "mean_uncert":    float(b_uncerts.mean()),
            },
            "BK": {
                "detection_rate": float(bk_detected.mean()),
                "ci95":           [lo_bk, hi_bk],
                "mean_prob":      float(bk_probs.mean()),
            },
        },
        "combined_auprc_B": comb_auprc,
        "combined_auroc_B": comb_auroc,
        "uncertainty": {
            "p25": float(uncert_pcts[0]), "p50": float(uncert_pcts[1]),
            "p75": float(uncert_pcts[2]), "p90": float(uncert_pcts[3]),
            "p95": float(uncert_pcts[4]),
            "high_uncert_rate": float(high_uncert),
        },
        "per_class":              per_class,
        "per_family_concurrent":  per_family_concurrent,
        "solo_baseline":          solo_stats,
        "concurrent_vs_solo":     deltas,
    }

    out_path = RESULTS_DIR / "mix_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
