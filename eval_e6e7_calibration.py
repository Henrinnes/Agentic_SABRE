#!/usr/bin/env python3
"""
eval_e6e7_calibration.py — E6 Calibration + E7 Conformal Triage
================================================================
NOT currently cited anywhere in TIFS3_Submission -- results/e6e7_calibration.json
exists (this script has been run at some point) but no number from it
appears in Main.tex or Supplementary.tex.

E6 — Calibration analysis for each agent before and after temperature scaling
  Metrics: ECE (15 equal-width bins), MCE, Brier score, reliability curve
  Calibration set: val split (fit temperature T on val, evaluate on test)
  Agents: [B] and [B+K] on RanSMAP ([B+K] uses a fixed corpus-mean
  Knowledge score, build_knowledge_lookup(), not a per-family lookup);
  [S] on RDset; [K] leave-one-out on RansKnow

E7 — Mondrian conformal prediction triage
  Inductive conformal prediction with class-conditional thresholds.
  Calibration nonconformity score for class y: s(x,y) = 1 - prob_y
    where prob_y = prob  if y=1,  else  1-prob
  Coverage levels: 1-ε ∈ {90%, 95%, 99%}
  Prediction set membership:
    include y=1 if 1 - prob ≤ q_1  (i.e. prob ≥ 1 - q_1)
    include y=0 if    prob  ≤ q_0
  Outcomes:
    DETECT    — set = {1}      (only ransomware)
    BENIGN    — set = {0}      (only benign)
    ESCALATE  — set = {0,1}    (uncertain, route to analyst)
    EMPTY     — set = {}       (should not occur at valid coverage)

Outputs
-------
  results/e6e7_calibration.json
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS, RANSKNOW_CSV

RANSMAP_PARQUET = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "ransmap_extracted/ransmap_train.parquet")
RDSET_PARQUET   = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "RDset/semantic_dataset_with_embeddings.parquet")
SPLITS_DIR      = Path("splits")
CKPT_DIR        = Path("checkpoints")
RESULTS_DIR     = Path("results")

SEED     = 42
T_MC     = 10
N_BINS   = 15    # ECE bins
COVERAGE = [0.90, 0.95, 0.99]

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP_COLS = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py

RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"}


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    bins  = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    ece_  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece_ += (mask.sum() / total) * abs(acc - conf)
    return float(ece_)


def mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    mce_ = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        mce_ = max(mce_, abs(acc - conf))
    return float(mce_)


def reliability_curve(probs: np.ndarray, labels: np.ndarray,
                      n_bins: int = N_BINS) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        curve.append({
            "bin_lo":    round(float(lo), 4),
            "bin_hi":    round(float(hi), 4),
            "n":         int(mask.sum()),
            "conf":      float(probs[mask].mean()) if mask.any() else None,
            "acc":       float(labels[mask].mean()) if mask.any() else None,
        })
    return curve


def calibration_metrics(probs: np.ndarray, labels: np.ndarray,
                         tag: str = "") -> dict:
    brier  = float(brier_score_loss(labels, probs))
    ece_   = ece(probs, labels)
    mce_   = mce(probs, labels)
    auroc  = float(roc_auc_score(labels, probs))
    auprc  = float(average_precision_score(labels, probs))
    curve  = reliability_curve(probs, labels)
    if tag:
        print(f"    {tag:<25}  ECE={ece_:.4f}  MCE={mce_:.4f}  "
              f"Brier={brier:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return {"ece": ece_, "mce": mce_, "brier": brier,
            "auroc": auroc, "auprc": auprc, "reliability": curve}


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(probs_val: np.ndarray, labels_val: np.ndarray) -> float:
    """Find T that minimises NLL on val set. logit(p)/T → sigmoid."""
    logits = logit(np.clip(probs_val, 1e-7, 1 - 1e-7))

    def nll(log_T: float) -> float:
        T   = np.exp(log_T)
        p_t = expit(logits / T)
        p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
        return -float((labels_val * np.log(p_t) +
                       (1 - labels_val) * np.log(1 - p_t)).mean())

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    T   = float(np.exp(res.x))
    return T


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = logit(np.clip(probs, 1e-7, 1 - 1e-7))
    return expit(logits / T).astype(np.float32)


# ---------------------------------------------------------------------------
# Conformal prediction (Mondrian / class-conditional)
# ---------------------------------------------------------------------------

def mondrian_thresholds(probs_cal: np.ndarray, labels_cal: np.ndarray,
                         epsilon: float) -> tuple[float, float]:
    """
    Inductive Mondrian conformal prediction.
    Nonconformity score s(x,y):
      y=1: s = 1 - prob        (miscalibrated high prob = low score)
      y=0: s = prob            (miscalibrated low prob  = low score)
    Returns (q1, q0): class-1 and class-0 quantile thresholds.
    """
    idx1 = labels_cal == 1
    idx0 = labels_cal == 0
    scores1 = 1.0 - probs_cal[idx1]   # smaller = better predicted as 1
    scores0 = probs_cal[idx0]          # smaller = better predicted as 0

    n1, n0 = idx1.sum(), idx0.sum()
    # Mondrian quantile at level ceil((n+1)*(1-ε))/n
    q1 = float(np.quantile(scores1, min(1.0, np.ceil((n1 + 1) * (1 - epsilon)) / n1)))
    q0 = float(np.quantile(scores0, min(1.0, np.ceil((n0 + 1) * (1 - epsilon)) / n0)))
    return q1, q0


def conformal_triage(probs_test: np.ndarray, labels_test: np.ndarray,
                      q1: float, q0: float) -> dict:
    """
    Apply conformal thresholds and measure triage outcomes.
    include_1: prob ≥ 1 - q1   (nonconformity score 1-prob ≤ q1)
    include_0: prob ≤ q0       (nonconformity score prob ≤ q0)
    """
    inc1 = (probs_test >= 1 - q1)
    inc0 = (probs_test <= q0)

    detect   = (inc1 & ~inc0).astype(int)   # only ransomware
    benign   = (inc0 & ~inc1).astype(int)   # only benign
    escalate = (inc1 & inc0).astype(int)    # both — uncertain
    empty    = (~inc1 & ~inc0).astype(int)  # neither

    n = len(probs_test)
    # Coverage: fraction of samples where true label is in prediction set
    y = labels_test.astype(int)
    correct = (y * inc1 + (1 - y) * inc0).astype(bool)
    coverage = float(correct.mean())

    # Precision/recall among DETECT decisions
    tp = int(((detect == 1) & (y == 1)).sum())
    fp = int(((detect == 1) & (y == 0)).sum())
    fn = int(((y == 1) & (detect == 0)).sum())
    det_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    det_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "detect_rate":    float(detect.mean()),
        "benign_rate":    float(benign.mean()),
        "escalate_rate":  float(escalate.mean()),
        "empty_rate":     float(empty.mean()),
        "empirical_coverage": coverage,
        "detect_precision":   det_precision,
        "detect_recall":      det_recall,
        "q1": q1, "q0": q0,
    }


# ---------------------------------------------------------------------------
# Data loaders and scorers
# ---------------------------------------------------------------------------

def load_ransmap_subset(subset: str, split: str = "random_stratified"):
    sess = set((SPLITS_DIR / split / f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP_PARQUET)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values
    return X, y, fam


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


def score_semantic(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from Agentic_SABRE.agents.semantic import SemanticAgent
    agent = SemanticAgent()
    outs  = agent.score_batch([{"embedding": emb[i]} for i in range(len(emb))])
    return (np.array([o.prob   for o in outs]),
            np.array([o.uncert for o in outs]))


def score_knowledge(X_know: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X_know[i]} for i in range(len(X_know))])
    return (np.array([o.prob   for o in outs]),
            np.array([o.uncert for o in outs]))


def build_knowledge_lookup():
    """Corpus-mean-always: identical Knowledge vector regardless of the
    window's true family. NOT currently called by any live paper number
    (this script's output is not cited in TIFS3_Submission), but a
    per-family lookup keyed by the sample's own true label -- as this
    function used to do -- is a label leak, matching the bug fixed in
    eval_sabre2.py, run_ea_ed_experiments.py, run_ee_eg_experiments.py,
    eval_mix_robustness.py, eval_e16_runtime.py, eval_e4_imbalance.py, and
    eval_e12_adversarial.py. Fixed here too so the leaky pattern isn't
    copy-pasted into future scripts from this one."""
    df_rk = pd.read_csv(RANSKNOW_CSV)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS, errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    corpus_mean = X_rk.mean(axis=0)
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)  # kept for API compatibility

    def lookup(cn: str) -> np.ndarray:
        return corpus_mean

    return lookup, zero_vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results: dict = {}

    # ===================================================================
    # E6-A: Behavioral Agent [B] — RanSMAP
    # ===================================================================
    print("\n" + "=" * 65)
    print("  E6-A: Behavioral Agent [B] — RanSMAP random_stratified")
    print("=" * 65)

    print("Scoring val set ...")
    X_val, y_val, _ = load_ransmap_subset("val")
    b_probs_val, _ = score_behavioral(X_val)

    print("Scoring test set ...")
    X_test, y_test, fam_test = load_ransmap_subset("test")
    b_probs_test, b_uncerts_test = score_behavioral(X_test)

    print("\n  Raw calibration:")
    m_raw_val  = calibration_metrics(b_probs_val,  y_val,  tag="[B] val  raw")
    m_raw_test = calibration_metrics(b_probs_test, y_test, tag="[B] test raw")

    T_b = fit_temperature(b_probs_val, y_val)
    print(f"\n  Temperature T fitted on val: {T_b:.4f}")

    b_cal_val  = apply_temperature(b_probs_val,  T_b)
    b_cal_test = apply_temperature(b_probs_test, T_b)

    print("\n  Temperature-scaled calibration:")
    m_cal_val  = calibration_metrics(b_cal_val,  y_val,  tag="[B] val  T-scaled")
    m_cal_test = calibration_metrics(b_cal_test, y_test, tag="[B] test T-scaled")

    results["behavioral"] = {
        "temperature": T_b,
        "val":  {"raw": m_raw_val,  "calibrated": m_cal_val},
        "test": {"raw": m_raw_test, "calibrated": m_cal_test},
    }

    # ===================================================================
    # E6-B: Knowledge Agent [K] — RansKnow (leave-one-channel-out)
    # ===================================================================
    print("\n" + "=" * 65)
    print("  E6-B: Knowledge Agent [K] — RansKnow LOCO")
    print("=" * 65)

    df_rk = pd.read_csv(RANSKNOW_CSV)
    df_rk["Ransomware_Context"] = (
        (df_rk["Tactic_Total_Mentions"] > 10) |
        (df_rk["Tool_Total_Mentions"]   > 2)
    ).astype(int)
    feat_cols_rk = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS + ["Ransomware_Context"], errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk  = df_rk[feat_cols_rk].fillna(0).to_numpy(dtype=np.float32)
    y_rk  = df_rk["Ransomware_Context"].astype(int).values
    grp_rk = df_rk["Channel_ID"].values

    from sklearn.model_selection import LeaveOneGroupOut
    logo  = LeaveOneGroupOut()
    k_all_probs, k_all_labels = [], []
    for train_idx, test_idx in logo.split(X_rk, y_rk, grp_rk):
        X_tr, y_tr = X_rk[train_idx], y_rk[train_idx]
        X_te, y_te = X_rk[test_idx],  y_rk[test_idx]
        k_probs_te, _ = score_knowledge(X_te)
        k_all_probs.extend(k_probs_te)
        k_all_labels.extend(y_te)

    k_all_probs  = np.array(k_all_probs,  dtype=np.float32)
    k_all_labels = np.array(k_all_labels, dtype=np.int32)

    print("\n  Raw calibration (LOCO pooled):")
    m_k = calibration_metrics(k_all_probs, k_all_labels, tag="[K] LOCO raw")

    T_k = fit_temperature(k_all_probs, k_all_labels)
    print(f"  Temperature T fitted: {T_k:.4f}")
    k_cal = apply_temperature(k_all_probs, T_k)
    print("\n  Temperature-scaled:")
    m_k_cal = calibration_metrics(k_cal, k_all_labels, tag="[K] LOCO T-scaled")

    results["knowledge"] = {
        "temperature": T_k,
        "loco": {"raw": m_k, "calibrated": m_k_cal},
    }

    # ===================================================================
    # E6-C: Semantic Agent [S] — RDset
    # ===================================================================
    print("\n" + "=" * 65)
    print("  E6-C: Semantic Agent [S] — RDset random split")
    print("=" * 65)

    df_rdset  = pd.read_parquet(RDSET_PARQUET)
    val_idx   = np.load(str(SPLITS_DIR / "rdset_random" / "val_indices.npy"))
    test_idx  = np.load(str(SPLITS_DIR / "rdset_random" / "test_indices.npy"))

    emb_val  = np.stack(df_rdset.iloc[val_idx]["embedding"].values).astype(np.float32)
    y_s_val  = df_rdset.iloc[val_idx]["label"].astype(int).values
    emb_test = np.stack(df_rdset.iloc[test_idx]["embedding"].values).astype(np.float32)
    y_s_test = df_rdset.iloc[test_idx]["label"].astype(int).values

    print(f"Scoring semantic val ({len(emb_val):,}) ...")
    s_probs_val, _ = score_semantic(emb_val)
    print(f"Scoring semantic test ({len(emb_test):,}) ...")
    s_probs_test, s_uncerts_test = score_semantic(emb_test)

    print("\n  Raw calibration:")
    m_s_raw_val  = calibration_metrics(s_probs_val,  y_s_val,  tag="[S] val  raw")
    m_s_raw_test = calibration_metrics(s_probs_test, y_s_test, tag="[S] test raw")

    T_s = fit_temperature(s_probs_val, y_s_val)
    print(f"\n  Temperature T: {T_s:.4f}")
    s_cal_val  = apply_temperature(s_probs_val,  T_s)
    s_cal_test = apply_temperature(s_probs_test, T_s)

    print("\n  Temperature-scaled:")
    m_s_cal_val  = calibration_metrics(s_cal_val,  y_s_val,  tag="[S] val  T-scaled")
    m_s_cal_test = calibration_metrics(s_cal_test, y_s_test, tag="[S] test T-scaled")

    results["semantic"] = {
        "temperature": T_s,
        "val":  {"raw": m_s_raw_val,  "calibrated": m_s_cal_val},
        "test": {"raw": m_s_raw_test, "calibrated": m_s_cal_test},
    }

    # ===================================================================
    # E6-D: Fused [B+K] on RanSMAP
    # ===================================================================
    print("\n" + "=" * 65)
    print("  E6-D: Fused [B+K] — RanSMAP random_stratified")
    print("=" * 65)

    kn_lookup, zero_vec = build_knowledge_lookup()

    X_know_val  = np.stack([kn_lookup(f) for f in load_ransmap_subset("val")[2]])
    X_know_test = np.stack([kn_lookup(f) for f in fam_test])

    print("Scoring knowledge val/test ...")
    k_probs_val,  _ = score_knowledge(X_know_val)
    k_probs_test, _ = score_knowledge(X_know_test)

    bk_probs_val  = (b_probs_val  + k_probs_val)  / 2.0
    bk_probs_test = (b_probs_test + k_probs_test) / 2.0

    print("\n  Raw calibration:")
    m_bk_raw_val  = calibration_metrics(bk_probs_val,  y_val,  tag="[B+K] val  raw")
    m_bk_raw_test = calibration_metrics(bk_probs_test, y_test, tag="[B+K] test raw")

    T_bk = fit_temperature(bk_probs_val, y_val)
    print(f"\n  Temperature T: {T_bk:.4f}")
    bk_cal_val  = apply_temperature(bk_probs_val,  T_bk)
    bk_cal_test = apply_temperature(bk_probs_test, T_bk)

    print("\n  Temperature-scaled:")
    m_bk_cal_val  = calibration_metrics(bk_cal_val,  y_val,  tag="[B+K] val  T-scaled")
    m_bk_cal_test = calibration_metrics(bk_cal_test, y_test, tag="[B+K] test T-scaled")

    results["fused_bk"] = {
        "temperature": T_bk,
        "val":  {"raw": m_bk_raw_val,  "calibrated": m_bk_cal_val},
        "test": {"raw": m_bk_raw_test, "calibrated": m_bk_cal_test},
    }

    # ===================================================================
    # E7: Mondrian Conformal Prediction Triage
    # ===================================================================
    print("\n" + "=" * 65)
    print("  E7: Mondrian Conformal Prediction Triage")
    print("=" * 65)

    conformal_results: dict = {}

    for agent_name, probs_cal, y_cal, probs_tst, y_tst in [
        ("[B]",    b_cal_val,  y_val,  b_cal_test,  y_test),
        ("[B+K]",  bk_cal_val, y_val,  bk_cal_test, y_test),
    ]:
        print(f"\n  Agent: {agent_name}")
        print(f"  {'Coverage':>9}  {'DETECT':>8}  {'BENIGN':>8}  "
              f"{'ESCALATE':>10}  {'Emp.Cov':>8}  {'Det.Prec':>9}  {'Det.Rec':>8}")
        print("  " + "-" * 72)

        agent_conf: dict = {}
        for cov in COVERAGE:
            eps = 1.0 - cov
            q1, q0 = mondrian_thresholds(probs_cal, y_cal, eps)
            triage  = conformal_triage(probs_tst, y_tst, q1, q0)
            agent_conf[str(cov)] = triage

            print(f"  {cov*100:>8.0f}%  "
                  f"{triage['detect_rate']:>8.3f}  "
                  f"{triage['benign_rate']:>8.3f}  "
                  f"{triage['escalate_rate']:>10.3f}  "
                  f"{triage['empirical_coverage']:>8.3f}  "
                  f"{triage['detect_precision']:>9.3f}  "
                  f"{triage['detect_recall']:>8.3f}")

        conformal_results[agent_name] = agent_conf

    results["conformal"] = conformal_results

    # ===================================================================
    # Summary table
    # ===================================================================
    print(f"\n{'='*70}")
    print("  E6/E7 SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Agent':<10} {'Set':<8}  {'ECE_raw':>8}  {'ECE_cal':>8}  "
          f"{'Brier_raw':>10}  {'Brier_cal':>10}  {'T':>6}")
    print("  " + "-" * 68)

    rows = [
        ("[B]",   "val",  results["behavioral"]["val"]["raw"],   results["behavioral"]["val"]["calibrated"],   T_b),
        ("[B]",   "test", results["behavioral"]["test"]["raw"],  results["behavioral"]["test"]["calibrated"],  T_b),
        ("[B+K]", "val",  results["fused_bk"]["val"]["raw"],     results["fused_bk"]["val"]["calibrated"],     T_bk),
        ("[B+K]", "test", results["fused_bk"]["test"]["raw"],    results["fused_bk"]["test"]["calibrated"],    T_bk),
        ("[S]",   "val",  results["semantic"]["val"]["raw"],     results["semantic"]["val"]["calibrated"],     T_s),
        ("[S]",   "test", results["semantic"]["test"]["raw"],    results["semantic"]["test"]["calibrated"],    T_s),
        ("[K]",   "loco", results["knowledge"]["loco"]["raw"],   results["knowledge"]["loco"]["calibrated"],   T_k),
    ]
    for agent, subset, raw, cal, T in rows:
        print(f"  {agent:<10} {subset:<8}  "
              f"{raw['ece']:>8.4f}  {cal['ece']:>8.4f}  "
              f"{raw['brier']:>10.4f}  {cal['brier']:>10.4f}  {T:>6.3f}")

    print(f"\n  Conformal triage ([B+K], calibrated probs):")
    print(f"  {'Coverage':>9}  {'DETECT':>8}  {'ESCALATE':>10}  {'EmpCov':>7}  {'Det.Prec':>9}")
    print("  " + "-" * 52)
    for cov in COVERAGE:
        t = conformal_results["[B+K]"][str(cov)]
        print(f"  {cov*100:>8.0f}%  {t['detect_rate']:>8.3f}  "
              f"{t['escalate_rate']:>10.3f}  {t['empirical_coverage']:>7.3f}  "
              f"{t['detect_precision']:>9.3f}")

    # ===================================================================
    # Save
    # ===================================================================
    out_path = RESULTS_DIR / "e6e7_calibration.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
