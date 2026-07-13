#!/usr/bin/env python3
"""
eval_e12_adversarial.py — E12 Adversarial Mimicry Stress Tests
===============================================================
NOT currently cited anywhere in TIFS3_Submission -- results/e12_adversarial.json
exists (this script has been run at some point) but no number from it
appears in Main.tex or Supplementary.tex. The "Adversarial robustness"
limitation paragraph in Main.tex's Discussion is a qualitative statement
that does not cite specific numbers from this script.

Simulates a knowledgeable adversary who perturbs ransomware behavioral
features to evade detection, and measures how well the uncertainty-based
ESCALATE route catches evasions that slip past prob ≥ 0.5.

Mimicry attack types
--------------------
  slow_encrypt   : Blend top-4 disk+read discriminators toward benign mean.
                   Targets: read_LBA_variance, mem_write_count,
                            mem_read_count, avg_read_throughput.
                   Simulates ransomware throttling its I/O and hiding
                   memory footprint (slow-encryption, chunked reads).

  mem_mask       : Blend all 10 memory features toward benign mean.
                   Simulates a process hollowing / LotL attack that hides
                   the in-memory encryption loop (mem_read/write/exec counts
                   and GPA variances made to look like a browser or IDE).

  full_blend     : Interpolate ALL 16 features toward the benign centroid.
                   Worst-case oracle evasion — adversary has full feature
                   knowledge and maximally obscures the ransomware profile.

Perturbation parameter α ∈ [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
  α = 0.0 → original ransomware features (no evasion)
  α = 1.0 → full blend to benign centroid (perfect camouflage)

Triage model
------------
  DETECT    : prob ≥ 0.50
  ESCALATE  : prob < 0.50  AND  uncert ≥ UNCERT_THRESH
  MISS      : prob < 0.50  AND  uncert < UNCERT_THRESH
  Triage coverage = DETECT + ESCALATE (windows not silently missed)

Agents evaluated
----------------
  [B]   : BehavioralAgent (random_stratified, T=10 MC-Dropout)
  [B+K] : [B] mean-fused with KnowledgeAgent's fixed corpus-mean score
          (build_knowledge_lookup() -- not a per-family lookup)

Outputs
-------
  results/e12_adversarial.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RANSMAP_PARQUET = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RANSKNOW_CSV    = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"
SPLITS_DIR      = Path("splits")
CKPT_DIR        = Path("checkpoints")
RESULTS_DIR     = Path("results")

SEED          = 42
T_MC          = 10
UNCERT_THRESH = 0.10   # uncertainty threshold for ESCALATE routing

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

FEATURE_IDX = {f: i for i, f in enumerate(FEATURE_COLS)}

# Feature groups for targeted mimicry
SLOW_ENCRYPT_FEATS = [      # top disk+read discriminators (scaled diff ≥ ±0.29)
    "read_LBA_variance",    # +0.595 (ransomware scattered reads)
    "mem_write_count",      # +0.505 (in-memory encryption writes)
    "mem_read_count",       # +0.453 (reads file content into memory)
    "avg_read_throughput",  # +0.295 (bulk read before encryption)
]

MEM_MASK_FEATS = [          # all memory-subsystem features
    "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean",
    "mem_read_GPA_variance", "mem_write_GPA_variance",
    "mem_readwrite_GPA_variance", "mem_exec_GPA_variance",
    "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_Count", "Family_List",
    "Tool_List", "Dominant_Tactic", "Platform_Signal",
]

RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"}

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Knowledge helpers
# ---------------------------------------------------------------------------

def build_knowledge_lookup() -> tuple[dict, np.ndarray, callable]:
    """Corpus-mean-always: identical Knowledge vector regardless of the
    window's true family. NOT currently called by any live paper number
    (this script's output is not cited in TIFS3_Submission), but a
    per-family lookup keyed by the sample's own true label -- as this
    function used to do -- is a label leak, matching the bug fixed in
    eval_sabre2.py, run_ea_ed_experiments.py, run_ee_eg_experiments.py,
    eval_mix_robustness.py, eval_e16_runtime.py, and eval_e4_imbalance.py.
    Fixed here too so the leaky pattern isn't copy-pasted into future
    scripts from this one."""
    df_rk = pd.read_csv(RANSKNOW_CSV)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    corpus_mean = X_rk.mean(axis=0)
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)  # kept for API compatibility

    def lookup(class_name: str) -> np.ndarray:
        return corpus_mean

    return {}, zero_vec, lookup


# ---------------------------------------------------------------------------
# Agent scorers
# ---------------------------------------------------------------------------

def score_behavioral(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / "behavioral_cnn_random_stratified.pt"),
        scaler_path=str(CKPT_DIR / "behavioral_scaler_random_stratified.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X[i]} for i in range(len(X))])
    return (np.array([o.prob   for o in outs]),
            np.array([o.uncert for o in outs]))


def score_knowledge(X_know: np.ndarray) -> np.ndarray:
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X_know[i]} for i in range(len(X_know))])
    return np.array([o.prob for o in outs])


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def perturb(X: np.ndarray, benign_centroid: np.ndarray,
            alpha: float, target_indices: list[int]) -> np.ndarray:
    """Blend target features toward benign centroid by alpha."""
    X_p = X.copy()
    for idx in target_indices:
        X_p[:, idx] = (1 - alpha) * X[:, idx] + alpha * benign_centroid[idx]
    return X_p


# ---------------------------------------------------------------------------
# Triage metrics
# ---------------------------------------------------------------------------

def triage_metrics(probs: np.ndarray, uncerts: np.ndarray,
                   uncert_thresh: float = UNCERT_THRESH) -> dict:
    detected  = (probs >= 0.50).astype(int)
    missed    = (probs < 0.50).astype(int)
    escalated = ((probs < 0.50) & (uncerts >= uncert_thresh)).astype(int)
    silently_missed = ((probs < 0.50) & (uncerts < uncert_thresh)).astype(int)

    n = len(probs)
    return {
        "detection_rate":    float(detected.mean()),
        "escalation_rate":   float(escalated.mean()),
        "triage_coverage":   float((detected | escalated).mean()),
        "miss_rate":         float(silently_missed.mean()),
        "mean_prob":         float(probs.mean()),
        "mean_uncert":       float(uncerts.mean()),
        "high_uncert_rate":  float((uncerts >= uncert_thresh).mean()),
    }


# ---------------------------------------------------------------------------
# Sweep one attack type
# ---------------------------------------------------------------------------

def sweep_attack(
    attack_name: str,
    X_ransom: np.ndarray,
    benign_centroid: np.ndarray,
    target_indices: list[int],
    X_know: np.ndarray,
) -> dict:
    results = {}
    print(f"\n  Attack: {attack_name}  |  target features: {len(target_indices)}")
    print(f"  {'α':>5}  {'DR_B':>6}  {'ER_B':>6}  {'TC_B':>6}  {'MR_B':>6}  "
          f"{'DR_BK':>6}  {'TC_BK':>6}  {'Uncert':>7}")
    print("  " + "-" * 68)

    for alpha in ALPHAS:
        X_p = perturb(X_ransom, benign_centroid, alpha, target_indices)

        b_probs, b_uncerts = score_behavioral(X_p)
        k_probs = score_knowledge(X_know)        # knowledge unchanged (static lookup)
        bk_probs = (b_probs + k_probs) / 2.0
        bk_uncerts = b_uncerts  # uncertainty from B only

        m_b  = triage_metrics(b_probs,  b_uncerts)
        m_bk = triage_metrics(bk_probs, bk_uncerts)

        results[str(alpha)] = {
            "alpha":  alpha,
            "B":  m_b,
            "BK": m_bk,
        }

        print(f"  {alpha:>5.1f}  {m_b['detection_rate']:>6.3f}  "
              f"{m_b['escalation_rate']:>6.3f}  {m_b['triage_coverage']:>6.3f}  "
              f"{m_b['miss_rate']:>6.3f}  "
              f"{m_bk['detection_rate']:>6.3f}  {m_bk['triage_coverage']:>6.3f}  "
              f"{m_b['mean_uncert']:>7.4f}")

    return results


# ---------------------------------------------------------------------------
# Per-family analysis at a fixed alpha
# ---------------------------------------------------------------------------

def family_breakdown(X_ransom: np.ndarray, families: np.ndarray,
                     benign_centroid: np.ndarray, target_indices: list[int],
                     X_know: np.ndarray, alpha: float) -> dict:
    X_p = perturb(X_ransom, benign_centroid, alpha, target_indices)
    b_probs, b_uncerts = score_behavioral(X_p)
    k_probs  = score_knowledge(X_know)
    bk_probs = (b_probs + k_probs) / 2.0

    out = {}
    for fam in sorted(np.unique(families)):
        mask = families == fam
        m_b  = triage_metrics(b_probs[mask],  b_uncerts[mask])
        m_bk = triage_metrics(bk_probs[mask], b_uncerts[mask])
        out[fam] = {"n": int(mask.sum()), "B": m_b, "BK": m_bk}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Load test split
    # -----------------------------------------------------------------------
    print("Loading random_stratified test split ...")
    test_sess = set(
        (SPLITS_DIR / "random_stratified" / "test_sessions.txt").read_text().splitlines()
    )
    df = pd.read_parquet(RANSMAP_PARQUET)
    df_test  = df[df["session"].isin(test_sess)].reset_index(drop=True)
    df_ransom = df_test[df_test["label"] == 1].reset_index(drop=True)
    df_benign = df_test[df_test["label"] == 0].reset_index(drop=True)

    X_ransom  = df_ransom[FEATURE_COLS].to_numpy(dtype=np.float32)
    families  = df_ransom["class_name"].values
    X_benign  = df_benign[FEATURE_COLS].to_numpy(dtype=np.float32)

    benign_centroid = X_benign.mean(axis=0)

    print(f"  Ransom windows: {len(X_ransom):,}  |  Benign centroid computed from {len(X_benign):,}")

    # -----------------------------------------------------------------------
    # Knowledge vectors (static — not perturbed)
    # -----------------------------------------------------------------------
    print("Building knowledge lookup (corpus-mean-always) ...")
    _, zero_vec, kn_lookup = build_knowledge_lookup()
    X_know = np.stack([kn_lookup(f) for f in families])
    print(f"  Knowledge feature dim: {X_know.shape[1]}  (identical vector for every window)")

    # -----------------------------------------------------------------------
    # Baseline (α = 0)
    # -----------------------------------------------------------------------
    print("\n=== Baseline (α=0, no perturbation) ===")
    b_probs0, b_uncerts0 = score_behavioral(X_ransom)
    k_probs0  = score_knowledge(X_know)
    bk_probs0 = (b_probs0 + k_probs0) / 2.0
    m0_b  = triage_metrics(b_probs0,  b_uncerts0)
    m0_bk = triage_metrics(bk_probs0, b_uncerts0)
    print(f"  [B]   DR={m0_b['detection_rate']:.3f}  TC={m0_b['triage_coverage']:.3f}  "
          f"MR={m0_b['miss_rate']:.3f}  uncert={m0_b['mean_uncert']:.4f}")
    print(f"  [B+K] DR={m0_bk['detection_rate']:.3f}  TC={m0_bk['triage_coverage']:.3f}  "
          f"MR={m0_bk['miss_rate']:.3f}")

    # -----------------------------------------------------------------------
    # Attack sweeps
    # -----------------------------------------------------------------------
    attack_configs = {
        "slow_encrypt": [FEATURE_IDX[f] for f in SLOW_ENCRYPT_FEATS],
        "mem_mask":     [FEATURE_IDX[f] for f in MEM_MASK_FEATS],
        "full_blend":   list(range(len(FEATURE_COLS))),
    }

    print("\n=== Attack sweeps (α from 0.0 to 1.0) ===")
    sweep_results = {}
    for attack_name, target_idx in attack_configs.items():
        sweep_results[attack_name] = sweep_attack(
            attack_name, X_ransom, benign_centroid, target_idx, X_know
        )

    # -----------------------------------------------------------------------
    # Per-family breakdown at mid-point alpha (0.5) for slow_encrypt
    # -----------------------------------------------------------------------
    BREAKDOWN_ALPHA = 0.5
    print(f"\n=== Per-family breakdown at α={BREAKDOWN_ALPHA} (slow_encrypt attack) ===")
    print(f"  {'Family':<16} {'N':>5}  {'DR_B':>6}  {'TC_B':>6}  {'MR_B':>6}  "
          f"{'DR_BK':>6}  {'TC_BK':>6}")
    print("  " + "-" * 62)

    fam_results = family_breakdown(
        X_ransom, families, benign_centroid,
        attack_configs["slow_encrypt"], X_know,
        alpha=BREAKDOWN_ALPHA,
    )
    for fam, r in sorted(fam_results.items()):
        print(f"  {fam:<16} {r['n']:>5}  "
              f"{r['B']['detection_rate']:>6.3f}  {r['B']['triage_coverage']:>6.3f}  "
              f"{r['B']['miss_rate']:>6.3f}  "
              f"{r['BK']['detection_rate']:>6.3f}  {r['BK']['triage_coverage']:>6.3f}")

    # -----------------------------------------------------------------------
    # Uncertainty-sweep: at what α does uncert ≥ UNCERT_THRESH?
    # -----------------------------------------------------------------------
    print(f"\n=== Uncertainty / triage coverage cross-over (slow_encrypt) ===")
    print(f"  α where DR_B < 0.50: find critical evasion threshold")
    slow_res = sweep_results["slow_encrypt"]
    for alpha_str, rv in slow_res.items():
        dr = rv["B"]["detection_rate"]
        tc = rv["B"]["triage_coverage"]
        hr = rv["B"]["high_uncert_rate"]
        mr = rv["B"]["miss_rate"]
        print(f"  α={float(alpha_str):.1f}  DR={dr:.3f}  TC={tc:.3f}  "
              f"high_uncert={hr:.3f}  miss={mr:.3f}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  E12 ADVERSARIAL MIMICRY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Attack':<14} {'α':>4}  {'DR_B':>6}  {'TC_B':>6}  {'MR_B':>6}  "
          f"{'DR_BK':>6}  {'TC_BK':>6}")
    print("  " + "-" * 60)
    for attack_name, res in sweep_results.items():
        for alpha_str, rv in res.items():
            alpha = float(alpha_str)
            if alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
                print(f"  {attack_name:<14} {alpha:>4.1f}  "
                      f"{rv['B']['detection_rate']:>6.3f}  "
                      f"{rv['B']['triage_coverage']:>6.3f}  "
                      f"{rv['B']['miss_rate']:>6.3f}  "
                      f"{rv['BK']['detection_rate']:>6.3f}  "
                      f"{rv['BK']['triage_coverage']:>6.3f}")
        print()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "baseline": {"B": m0_b, "BK": m0_bk},
        "attacks":  sweep_results,
        "per_family_slow_encrypt_alpha05": fam_results,
        "config": {
            "uncert_thresh":   UNCERT_THRESH,
            "T_mc":            T_MC,
            "alphas":          ALPHAS,
            "slow_encrypt_feats": SLOW_ENCRYPT_FEATS,
            "mem_mask_feats":     MEM_MASK_FEATS,
        },
    }

    out_path = RESULTS_DIR / "e12_adversarial.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
