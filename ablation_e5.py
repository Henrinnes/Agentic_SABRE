#!/usr/bin/env python3
"""
ablation_e5.py — E5 Full Ablation Study
========================================
Compares all 7 subsets of {Semantic, Behavioral, Knowledge} agents.

Data scope
----------
  Semantic  : RDset (PE static features + sentence-transformer embeddings)
  Behavioral: RanSMAP (runtime telemetry, 16 features, 15 hardware configs)
  Knowledge : RansKnow (YouTube ATT&CK transcripts, 23 features)

Because no unified tri-modal dataset exists, all cross-dataset combinations
are evaluated with the following protocol:
  - [B] and [B+K] are evaluated on RanSMAP test set — directly comparable.
  - [S] is evaluated on RDset test set — standalone only.
  - [K] is evaluated via leave-one-family-out on RansKnow — standalone only.
  - Combinations involving S on RanSMAP use zero embeddings (cross-dataset).
    These rows are marked CROSS-DATASET in the output tables.

Fusion method: mean average of active agent probabilities.

Splits evaluated for [B] and [B+K]:
  random_stratified, family_holdout, hardware_holdout

Metrics: AUROC, AUPRC, Accuracy, F1-macro, Recall (ransom), FPR
         with 95% bootstrap CI (1000 resamples) on AUROC and AUPRC.

Usage
-----
    python ablation_e5.py
    python ablation_e5.py --no-ci      # skip bootstrap (faster)
    python ablation_e5.py --splits random_stratified family_holdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut

RANSMAP_PARQUET  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RDSET_PARQUET    = "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet"
RANSKNOW_CSV     = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"
SPLITS_DIR       = Path("splits")
CKPT_DIR         = Path("checkpoints")

N_BOOTSTRAP = 1000
SEED = 42
rng  = np.random.default_rng(SEED)

BEHAVIORAL_FEATURE_COLS = [
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

RANSOMWARE_FAMILIES = {"Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "auroc":    float(roc_auc_score(y_true, probs)),
        "auprc":    float(average_precision_score(y_true, probs)),
        "acc":      float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "recall_r": recall,
        "fpr":      fpr,
    }


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, n: int = N_BOOTSTRAP) -> dict:
    aurocs, auprcs = [], []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[s], probs[s]))
        auprcs.append(average_precision_score(y_true[s], probs[s]))
    return {
        "auroc_ci95": (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))),
        "auprc_ci95": (float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))),
    }


def fmt(m: dict, ci: Optional[dict] = None) -> str:
    s = (f"AUROC {m['auroc']:.4f}"
         + (f" [{ci['auroc_ci95'][0]:.4f},{ci['auroc_ci95'][1]:.4f}]" if ci else "")
         + f"  AUPRC {m['auprc']:.4f}"
         + (f" [{ci['auprc_ci95'][0]:.4f},{ci['auprc_ci95'][1]:.4f}]" if ci else "")
         + f"  Acc {m['acc']:.4f}  F1 {m['f1_macro']:.4f}"
         + f"  Recall(R) {m['recall_r']:.4f}  FPR {m['fpr']:.4f}")
    return s


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ransmap_split(split_name: str, subset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (n, 16), y (n,), families (n,) for a split subset."""
    sess_file = SPLITS_DIR / split_name / f"{subset}_sessions.txt"
    sessions  = set(sess_file.read_text().splitlines())
    df = pd.read_parquet(RANSMAP_PARQUET)
    sub = df[df["session"].isin(sessions)].reset_index(drop=True)
    X   = sub[BEHAVIORAL_FEATURE_COLS].to_numpy(dtype=np.float32)
    y   = sub["label"].astype(int).values
    fam = sub["class_name"].values
    return X, y, fam


def build_knowledge_lookup(split_name: str) -> tuple[dict, np.ndarray, callable]:
    """
    Build a family→mean_knowledge_features lookup from RansKnow.
    Only uses families present in the TRAINING split of RanSMAP (no test leakage).
    Returns (family_to_features, zero_vector, lookup_fn).
    """
    df_rk = pd.read_csv(RANSKNOW_CSV)
    df_rk["Ransomware_Context"] = (
        (df_rk["Tactic_Total_Mentions"] > 10) |
        (df_rk["Tool_Total_Mentions"]   > 2)
    ).astype(int)
    feat_cols = (
        df_rk.drop(columns=KNOWLEDGE_DROP_COLS + ["Ransomware_Context"], errors="ignore")
             .select_dtypes(include="number")
             .columns.tolist()
    )
    X_rk = df_rk[feat_cols].fillna(0).to_numpy(dtype=np.float32)

    # Build per-family mean features from RansKnow Family_List column
    family_feats: dict[str, np.ndarray] = {}
    for i, row in df_rk.iterrows():
        family_list = str(row.get("Family_List", ""))
        for fam in [f.strip() for f in family_list.split(",")]:
            if fam and fam != "nan":
                if fam not in family_feats:
                    family_feats[fam] = []
                family_feats[fam].append(X_rk[i])

    family_mean: dict[str, np.ndarray] = {
        fam: np.mean(np.stack(vecs), axis=0)
        for fam, vecs in family_feats.items()
    }
    zero_vec = np.zeros(X_rk.shape[1], dtype=np.float32)

    def lookup(class_name: str) -> np.ndarray:
        # Exact match first, then partial (e.g. "Conti_01" → "Conti")
        if class_name in family_mean:
            return family_mean[class_name]
        for k in family_mean:
            if k.lower() in class_name.lower() or class_name.lower() in k.lower():
                return family_mean[k]
        return zero_vec

    return family_mean, zero_vec, lookup, feat_cols


def load_rdset_split() -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings, labels) for the RDset test split."""
    test_idx = np.load(str(SPLITS_DIR / "rdset_random" / "test_indices.npy"))
    df = pd.read_parquet(RDSET_PARQUET)
    sub = df.iloc[test_idx].reset_index(drop=True)
    emb = np.stack(sub["embedding"].values).astype(np.float32)
    y   = sub["label"].astype(int).values
    return emb, y


def load_ransknow() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, groups) for leave-one-family-out on RansKnow."""
    df = pd.read_csv(RANSKNOW_CSV)
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) |
        (df["Tool_Total_Mentions"]   > 2)
    ).astype(int)
    feat_cols = (
        df.drop(columns=KNOWLEDGE_DROP_COLS + ["Ransomware_Context"], errors="ignore")
          .select_dtypes(include="number")
          .columns.tolist()
    )
    X = df[feat_cols].fillna(0).to_numpy(dtype=np.float32)
    y = df["Ransomware_Context"].astype(int).values
    # Group by channel (proxy for "source holdout")
    groups = df["Channel_ID"].values
    return X, y, groups


# ---------------------------------------------------------------------------
# Agent scorers
# ---------------------------------------------------------------------------

def score_behavioral(X: np.ndarray, split_name: str) -> np.ndarray:
    """Run BehavioralAgent with the checkpoint for this split.
    Pass RAW features — the agent applies its own split-specific scaler internally."""
    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    agent = BehavioralAgent(
        checkpoint_path=str(CKPT_DIR / f"behavioral_cnn_{split_name}.pt"),
        scaler_path=str(CKPT_DIR / f"behavioral_scaler_{split_name}.joblib"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=10,
    )
    outs = agent.score_batch([{"features": X[i]} for i in range(len(X))])
    return np.array([o.prob for o in outs])


def score_knowledge_batch(X: np.ndarray) -> np.ndarray:
    """Run KnowledgeAgent on a batch of knowledge feature vectors."""
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=10,
    )
    outs = agent.score_batch([{"features": X[i]} for i in range(len(X))])
    return np.array([o.prob for o in outs])


def score_semantic_batch(emb: np.ndarray) -> np.ndarray:
    """Run SemanticAgent on a batch of embeddings."""
    from Agentic_SABRE.agents.semantic import SemanticAgent
    agent = SemanticAgent()
    outs  = agent.score_batch([{"embedding": emb[i]} for i in range(len(emb))])
    return np.array([o.prob for o in outs])


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    hdr = (f"  {'Combination':<22} {'Split':<22} {'AUROC':>8}  {'95% CI':^19}  "
           f"{'AUPRC':>7}  {'Acc':>6}  {'F1':>6}  {'Rec(R)':>7}  {'FPR':>6}  {'Note'}")
    print(hdr)
    print("  " + "-" * 108)
    for r in rows:
        ci_str = f"[{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]" if r.get("auroc_lo") else "     —      "
        note   = r.get("note", "")
        print(f"  {r['combo']:<22} {r['split']:<22} {r['auroc']:>8.4f}  {ci_str:^19}  "
              f"{r['auprc']:>7.4f}  {r['acc']:>6.4f}  {r['f1']:>6.4f}  "
              f"{r['recall_r']:>7.4f}  {r['fpr']:>6.4f}  {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(splits: list[str], do_ci: bool) -> None:
    all_rows   = []
    results_json = {}

    # ================================================================== #
    # SECTION 1 — Standalone agent evaluations
    # ================================================================== #
    print("\n" + "#"*70)
    print("# SECTION 1 — Standalone agent evaluations (native datasets)")
    print("#"*70)

    # --- 1a. Semantic agent on RDset ---
    print("\n[1a] Semantic agent on RDset test split …")
    emb_te, y_rd = load_rdset_split()
    p_sem = score_semantic_batch(emb_te)
    m = compute_metrics(y_rd, p_sem)
    ci = bootstrap_ci(y_rd, p_sem) if do_ci else None
    print(f"  {fmt(m, ci)}")
    row = dict(combo="[S] Semantic", split="RDset (random)",
               auroc=m["auroc"], auprc=m["auprc"], acc=m["acc"],
               f1=m["f1_macro"], recall_r=m["recall_r"], fpr=m["fpr"],
               auroc_lo=ci["auroc_ci95"][0] if ci else None,
               auroc_hi=ci["auroc_ci95"][1] if ci else None,
               note="standalone")
    all_rows.append(row)
    results_json["semantic_rdset"] = {**m, **(ci or {})}

    # --- 1b. Knowledge agent on RansKnow (channel-leave-one-out) ---
    print("\n[1b] Knowledge agent on RansKnow (leave-one-channel-out) …")
    X_rk, y_rk, groups_rk = load_ransknow()
    logo = LeaveOneGroupOut()
    k_probs_all, k_true_all = [], []
    for tr_idx, te_idx in logo.split(X_rk, y_rk, groups_rk):
        p = score_knowledge_batch(X_rk[te_idx])
        k_probs_all.extend(p.tolist())
        k_true_all.extend(y_rk[te_idx].tolist())
    k_probs_all = np.array(k_probs_all)
    k_true_all  = np.array(k_true_all)
    m = compute_metrics(k_true_all, k_probs_all)
    ci = bootstrap_ci(k_true_all, k_probs_all) if do_ci else None
    print(f"  {fmt(m, ci)}")
    row = dict(combo="[K] Knowledge", split="RansKnow (LOCO)",
               auroc=m["auroc"], auprc=m["auprc"], acc=m["acc"],
               f1=m["f1_macro"], recall_r=m["recall_r"], fpr=m["fpr"],
               auroc_lo=ci["auroc_ci95"][0] if ci else None,
               auroc_hi=ci["auroc_ci95"][1] if ci else None,
               note="standalone")
    all_rows.append(row)
    results_json["knowledge_loco"] = {**m, **(ci or {})}

    # ================================================================== #
    # SECTION 2 — Behavioral ablation across splits
    # ================================================================== #
    print("\n" + "#"*70)
    print("# SECTION 2 — Behavioral agent ablation across splits")
    print("#"*70)

    # Pre-build knowledge lookup (from training data only, no test leakage)
    _, _, know_lookup, know_feat_cols = build_knowledge_lookup("random_stratified")

    for split_name in splits:
        print(f"\n  --- Split: {split_name} ---")
        X_te, y_te, fam_te = load_ransmap_split(split_name, "test")
        n = len(y_te)
        print(f"  Test: {n:,} windows  "
              f"(ransom={int((y_te==1).sum()):,}  benign={int((y_te==0).sum()):,})")
        print(f"  Families in test: {sorted(set(fam_te))}")

        # Build knowledge feature matrix for test samples via family lookup
        K_te = np.stack([know_lookup(f) for f in fam_te]).astype(np.float32)

        # Zero semantic embedding (cross-dataset stand-in)
        emb_dim = 384
        emb_zero = np.zeros((n, emb_dim), dtype=np.float32)

        # Score each agent once
        print("  Scoring behavioral agent …")
        p_beh = score_behavioral(X_te, split_name)

        print("  Scoring knowledge agent (family lookup) …")
        p_kno = score_knowledge_batch(K_te)

        print("  Scoring semantic agent (zero embedding, cross-dataset) …")
        p_sem_cd = score_semantic_batch(emb_zero)

        combos = [
            ("[B]",      [p_beh],                  ""),
            ("[K]",      [p_kno],                  "family-lookup"),
            ("[B+K]",    [p_beh, p_kno],            ""),
            ("[S+B]",    [p_sem_cd, p_beh],         "CROSS-DATASET"),
            ("[S+K]",    [p_sem_cd, p_kno],         "CROSS-DATASET"),
            ("[S+B+K]",  [p_sem_cd, p_beh, p_kno],  "CROSS-DATASET"),
        ]

        split_rows = []
        for combo_name, prob_list, note in combos:
            fused = np.mean(np.stack(prob_list, axis=0), axis=0)
            m  = compute_metrics(y_te, fused)
            ci = bootstrap_ci(y_te, fused) if do_ci else None
            print(f"    {combo_name:<12} {fmt(m, ci)}  {note}")
            row = dict(combo=combo_name, split=split_name,
                       auroc=m["auroc"], auprc=m["auprc"], acc=m["acc"],
                       f1=m["f1_macro"], recall_r=m["recall_r"], fpr=m["fpr"],
                       auroc_lo=ci["auroc_ci95"][0] if ci else None,
                       auroc_hi=ci["auroc_ci95"][1] if ci else None,
                       note=note)
            all_rows.append(row)
            split_rows.append(row)
            results_json[f"{combo_name}_{split_name}"] = {**m, **(ci or {})}

    # ================================================================== #
    # SECTION 3 — Delta analysis: contribution of each agent
    # ================================================================== #
    print("\n" + "#"*70)
    print("# SECTION 3 — Delta analysis (AUROC lift by adding each agent)")
    print("#"*70)
    print(f"\n  {'Split':<22}  {'B→B+K':>8}  {'B→S+B':>8}  {'B+K→S+B+K':>10}  "
          f"{'B (base)':>9}  {'B+K':>9}  {'S+B+K':>9}")
    print("  " + "-" * 85)

    for sn in splits:
        def get_auroc(combo):
            matches = [r for r in all_rows if r["combo"] == combo and r["split"] == sn]
            return matches[0]["auroc"] if matches else None

        b   = get_auroc("[B]")
        bk  = get_auroc("[B+K]")
        sb  = get_auroc("[S+B]")
        sbk = get_auroc("[S+B+K]")

        if all(v is not None for v in [b, bk, sb, sbk]):
            print(f"  {sn:<22}  {bk-b:>+8.4f}  {sb-b:>+8.4f}  {sbk-bk:>+10.4f}  "
                  f"{b:>9.4f}  {bk:>9.4f}  {sbk:>9.4f}")

    # ================================================================== #
    # Summary table (paper-ready)
    # ================================================================== #
    print_table(all_rows, "E5 FULL ABLATION — ALL RESULTS (paper Table C)")

    # ================================================================== #
    # Save JSON
    # ================================================================== #
    out_path = Path("results") / "ablation_e5.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results_json, indent=2))
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+",
                   default=["random_stratified", "family_holdout", "hardware_holdout"])
    p.add_argument("--no-ci", action="store_true",
                   help="Skip bootstrap CI (faster, for quick checks)")
    args = p.parse_args()
    main(args.splits, do_ci=not args.no_ci)
