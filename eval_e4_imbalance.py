#!/usr/bin/env python3
"""
eval_e4_imbalance.py — E4 Realistic Imbalance Evaluation
==========================================================
NOT currently cited anywhere in TIFS3_Submission -- results/e4_imbalance.json
exists (this script has been run at some point) but no number from it
appears in Main.tex or Supplementary.tex.

Enterprise endpoints see ~95% benign / 5% ransomware traffic.
Standard 50/50 test sets overstate recall and understate FP cost.

Protocol
--------
For each split's test set:
  1. Resample to 5% ransomware / 95% benign (downsample ransomware).
     Repeat N_REPS=100 times with different seeds for stable estimates.
  2. Evaluate [B] and [B+K] on each resample. [B+K] uses a fixed
     corpus-mean Knowledge score (build_knowledge_lookup()), not a
     per-family lookup -- see that function's docstring.
  3. Report: AUPRC, PR-curve points, FP/day and FN/day at 10K events/day,
     cost-weighted risk, optimal threshold under realistic cost ratio.

Cost model
----------
  c_FP = 1   (analyst review, ~5 min)
  c_FN = 20  (missed ransomware: 20× more expensive than a false alarm)
  optimal threshold = argmin_t  c_FP * FPR(t) + c_FN * FNR(t)

Outputs
-------
  results/e4_imbalance.json
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS, RANSKNOW_CSV

RANSMAP_PARQUET = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "ransmap_extracted/ransmap_train.parquet")
SPLITS_DIR      = Path("splits")
CKPT_DIR        = Path("checkpoints")
RESULTS_DIR     = Path("results")

RANSOM_RATE   = 0.05          # 5% enterprise ransomware base rate
EVENTS_PER_DAY = 10_000       # windows processed per endpoint per day
N_REPS        = 100           # resampling repetitions for stable estimates
C_FP          = 1             # cost weight for false positive
C_FN          = 20            # cost weight for false negative (missed ransomware)
SEED          = 42

BEHAVIORAL_FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP_COLS = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py

SPLITS = ["random_stratified", "family_holdout", "hardware_holdout", "temporal_holdout"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_windows(split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sess = set((SPLITS_DIR / split_name / "test_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP_PARQUET)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[BEHAVIORAL_FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values
    return X, y, fam


def build_knowledge_lookup():
    """Corpus-mean-always: identical Knowledge vector regardless of the
    window's true family. NOT currently called by any live paper number
    (this script's output is not cited in TIFS3_Submission), but a
    per-family lookup keyed by the sample's own true label -- as this
    function used to do -- is a label leak, matching the bug fixed in
    eval_sabre2.py, run_ea_ed_experiments.py, run_ee_eg_experiments.py,
    eval_mix_robustness.py, and eval_e16_runtime.py. Fixed here too so the
    leaky pattern isn't copy-pasted into future scripts from this one."""
    df_rk = pd.read_csv(RANSKNOW_CSV)
    df_rk["Ransomware_Context"] = (
        (df_rk["Tactic_Total_Mentions"] > 10) | (df_rk["Tool_Total_Mentions"] > 2)
    ).astype(int)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS + ["Ransomware_Context"], errors="ignore")
             .select_dtypes(include="number").columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    corpus_mean = X_rk.mean(axis=0)

    def lookup(cn: str) -> np.ndarray:
        return corpus_mean

    return lookup


def get_probs(X: np.ndarray, fam: np.ndarray, split_name: str,
              know_lookup) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_b, p_bk) — raw behavioral and B+K fused probabilities."""
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    from Agentic_SABRE.agents.knowledge   import KnowledgeAgent

    beh_agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / f"behavioral_cnn_{split_name}.pt"),
        scaler_path=str(CKPT_DIR / f"behavioral_scaler_{split_name}.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=10,
    )
    kno_agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=10,
    )

    p_b   = np.array([o.prob for o in beh_agent.score_batch(
                [{"features": X[i]} for i in range(len(X))])])
    K_te  = np.stack([know_lookup(f) for f in fam]).astype(np.float32)
    p_k   = np.array([o.prob for o in kno_agent.score_batch(
                [{"features": K_te[i]} for i in range(len(K_te))])])
    p_bk  = (p_b + p_k) / 2.0
    return p_b, p_bk


def optimal_threshold(y: np.ndarray, probs: np.ndarray,
                      c_fp: float, c_fn: float) -> tuple[float, float]:
    """Return (threshold, cost) that minimises c_fp*FPR + c_fn*FNR."""
    thresholds = np.linspace(0, 1, 201)
    best_t, best_cost = 0.5, float("inf")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    for t in thresholds:
        preds = (probs >= t).astype(int)
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = fn / n_pos if n_pos > 0 else 0.0
        cost = c_fp * fpr + c_fn * fnr
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, best_cost


def imbalanced_metrics(y: np.ndarray, probs: np.ndarray,
                       threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    recall   = tp / n_pos if n_pos > 0 else 0.0
    fpr      = fp / n_neg if n_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Scale to EVENTS_PER_DAY
    ransom_per_day = EVENTS_PER_DAY * RANSOM_RATE       # 500
    benign_per_day = EVENTS_PER_DAY * (1 - RANSOM_RATE) # 9500
    fp_per_day = fpr    * benign_per_day
    fn_per_day = (1 - recall) * ransom_per_day
    return {
        "auprc":       float(average_precision_score(y, probs)),
        "auroc":       float(roc_auc_score(y, probs)),
        "threshold":   threshold,
        "recall":      recall,
        "fpr":         fpr,
        "precision":   precision,
        "fp_per_day":  fp_per_day,
        "fn_per_day":  fn_per_day,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pr_curve_points(y: np.ndarray, probs: np.ndarray,
                    recall_targets=(0.80, 0.85, 0.90, 0.95)) -> dict:
    """Precision at fixed recall targets."""
    prec, rec, _ = precision_recall_curve(y, probs)
    out = {}
    for rt in recall_targets:
        # Find highest precision where recall >= rt
        mask = rec >= rt
        out[f"prec@rec{int(rt*100)}"] = float(prec[mask].max()) if mask.any() else 0.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rng         = np.random.default_rng(SEED)
    know_lookup = build_knowledge_lookup()
    all_results = {}

    print(f"E4 Realistic Imbalance  —  {RANSOM_RATE*100:.0f}% ransomware / "
          f"{(1-RANSOM_RATE)*100:.0f}% benign  |  {N_REPS} resamples  |  "
          f"{EVENTS_PER_DAY:,} events/day")
    print(f"Cost model: c_FP={C_FP}, c_FN={C_FN}\n")

    hdr = (f"  {'Split':<22} {'Combo':<8} {'AUPRC':>8}  {'±':>6}  "
           f"{'AUROC':>8}  {'Recall':>7}  {'Prec':>7}  "
           f"{'FP/day':>8}  {'FN/day':>8}  {'P@R90':>7}")
    sep = "  " + "-" * 100

    print(hdr)
    print(sep)

    for split_name in SPLITS:
        X_te, y_te, fam_te = load_test_windows(split_name)
        n_ransom = int((y_te == 1).sum())
        n_benign = int((y_te == 0).sum())
        print(f"\n  [{split_name}]  test pool: {n_ransom} ransom / {n_benign} benign")

        # Score all test windows once with each model
        print(f"    Scoring agents …")
        p_b, p_bk = get_probs(X_te, fam_te, split_name, know_lookup)

        # How many ransomware to keep per resample for RANSOM_RATE
        # n_ransom_sub / (n_ransom_sub + n_benign) = RANSOM_RATE
        # → n_ransom_sub = n_benign * RANSOM_RATE / (1 - RANSOM_RATE)
        n_ransom_sub = max(1, int(n_benign * RANSOM_RATE / (1 - RANSOM_RATE)))
        n_ransom_sub = min(n_ransom_sub, n_ransom)  # can't exceed available
        actual_rate  = n_ransom_sub / (n_ransom_sub + n_benign)
        print(f"    Resample: {n_ransom_sub} ransom + {n_benign} benign "
              f"= {n_ransom_sub + n_benign} windows  "
              f"(actual rate {actual_rate*100:.1f}%)")

        ransom_idx = np.where(y_te == 1)[0]
        benign_idx = np.where(y_te == 0)[0]

        for combo_name, probs_full in [("[B]", p_b), ("[B+K]", p_bk)]:
            auprc_reps, auroc_reps = [], []
            rec_reps, prec_reps    = [], []
            fp_day_reps, fn_day_reps = [], []
            pr90_reps              = []

            for rep in range(N_REPS):
                sub_r = rng.choice(ransom_idx, size=n_ransom_sub, replace=False)
                idx   = np.concatenate([sub_r, benign_idx])
                y_sub = y_te[idx]
                p_sub = probs_full[idx]

                opt_t, _ = optimal_threshold(y_sub, p_sub, C_FP, C_FN)
                m = imbalanced_metrics(y_sub, p_sub, opt_t)
                pr = pr_curve_points(y_sub, p_sub)

                auprc_reps.append(m["auprc"])
                auroc_reps.append(m["auroc"])
                rec_reps.append(m["recall"])
                prec_reps.append(m["precision"])
                fp_day_reps.append(m["fp_per_day"])
                fn_day_reps.append(m["fn_per_day"])
                pr90_reps.append(pr.get("prec@rec90", 0.0))

            def ms(v): return float(np.mean(v)), float(np.std(v))

            auprc_m, auprc_s = ms(auprc_reps)
            auroc_m, auroc_s = ms(auroc_reps)
            rec_m,   _       = ms(rec_reps)
            prec_m,  _       = ms(prec_reps)
            fp_m,    fp_s    = ms(fp_day_reps)
            fn_m,    fn_s    = ms(fn_day_reps)
            pr90_m,  _       = ms(pr90_reps)

            print(f"  {split_name:<22} {combo_name:<8} "
                  f"{auprc_m:>8.4f}  {auprc_s:>6.4f}  "
                  f"{auroc_m:>8.4f}  {rec_m:>7.4f}  {prec_m:>7.4f}  "
                  f"{fp_m:>8.1f}  {fn_m:>8.1f}  {pr90_m:>7.4f}")

            all_results[f"{split_name}_{combo_name}"] = {
                "split": split_name, "combo": combo_name,
                "n_ransom_sub": n_ransom_sub, "n_benign": n_benign,
                "actual_ransom_rate": round(actual_rate, 4),
                "auprc_mean": round(auprc_m, 4), "auprc_std": round(auprc_s, 4),
                "auroc_mean": round(auroc_m, 4), "auroc_std": round(auroc_s, 4),
                "recall_mean": round(rec_m, 4),
                "precision_mean": round(prec_m, 4),
                "fp_per_day_mean": round(fp_m, 2), "fp_per_day_std": round(fp_s, 2),
                "fn_per_day_mean": round(fn_m, 2), "fn_per_day_std": round(fn_s, 2),
                "prec_at_rec90_mean": round(pr90_m, 4),
                "cost_model": {"c_fp": C_FP, "c_fn": C_FN,
                               "events_per_day": EVENTS_PER_DAY},
            }

    # Delta table
    print(f"\n{'='*80}")
    print("  Delta [B+K] vs [B]  (imbalanced, mean over 100 resamples)")
    print(f"{'='*80}")
    print(f"  {'Split':<22}  {'ΔAUPRC':>8}  {'ΔAUROC':>8}  "
          f"{'ΔFP/day':>9}  {'ΔFN/day':>9}")
    print("  " + "-" * 65)
    for sn in SPLITS:
        b   = all_results.get(f"{sn}_[B]")
        bk  = all_results.get(f"{sn}_[B+K]")
        if b and bk:
            print(f"  {sn:<22}  "
                  f"{bk['auprc_mean']-b['auprc_mean']:>+8.4f}  "
                  f"{bk['auroc_mean']-b['auroc_mean']:>+8.4f}  "
                  f"{bk['fp_per_day_mean']-b['fp_per_day_mean']:>+9.1f}  "
                  f"{bk['fn_per_day_mean']-b['fn_per_day_mean']:>+9.1f}")

    print(f"\n  Assumptions: {EVENTS_PER_DAY:,} events/day, "
          f"{RANSOM_RATE*100:.0f}% ransomware base rate, "
          f"cost ratio c_FN/c_FP = {C_FN}/{C_FP}")

    out = RESULTS_DIR / "e4_imbalance.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    main()
