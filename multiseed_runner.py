#!/usr/bin/env python3
"""
multiseed_runner.py — Multi-seed training and evaluation for statistical rigor.
================================================================================
Trains behavioral CNN across 5 seeds (0–4) × 4 data splits.
Seed 42 (canonical) results are loaded from checkpoints/metadata.json.

For each (seed, split):
  - Retrains behavioral CNN with that seed's SMOTE + weight init
  - Evaluates test AUROC, AUPRC, Accuracy on frozen test set
  - For random_stratified: also evaluates [B+K] fusion

Summary:
  - Mean ± Std over 6 seeds (0,1,2,3,4,42)
  - 95% CI via two-sided t-distribution: x̄ ± t_{0.025,n-1} · s / √n

Outputs
-------
  checkpoints/seeds/seed_{s}_{split}.pt
  checkpoints/seeds/seed_{s}_{split}_scaler.joblib
  results/multiseed.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, average_precision_score,
                              roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PARQUET    = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RANSKNOW   = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_718.csv"
SPLITS_DIR = Path("splits")
CKPT_DIR   = Path("checkpoints")
SEED_DIR   = CKPT_DIR / "seeds"
RESULTS    = Path("results")

EPOCHS   = 30
BATCH    = 256
LR       = 1e-3
DROPOUT  = 0.3
PATIENCE = 7
T_MC     = 10

SPLITS   = ["random_stratified", "family_holdout",
             "hardware_holdout", "temporal_holdout"]
SEEDS    = [0, 1, 2, 3, 4]   # seed 42 loaded from existing metadata

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
# Model (identical to retrain_behavioural.py)
# ---------------------------------------------------------------------------
class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim: int = 16, dropout: float = DROPOUT):
        super().__init__()
        pooled = (input_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Flatten(), nn.Linear(pooled * 16, 64),
            nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_sessions(split: str, subset: str) -> set:
    return set((SPLITS_DIR / split / f"{subset}_sessions.txt")
               .read_text().splitlines())


def load_split_data(df: pd.DataFrame, split: str):
    tr = load_sessions(split, "train")
    vl = load_sessions(split, "val")
    te = load_sessions(split, "test")
    d_tr = df[df["session"].isin(tr)].reset_index(drop=True)
    d_vl = df[df["session"].isin(vl)].reset_index(drop=True)
    d_te = df[df["session"].isin(te)].reset_index(drop=True)
    return d_tr, d_vl, d_te


def evaluate_model(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[float, float, float]:
    model.eval()
    crit = nn.BCELoss()
    all_probs, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            total_loss += crit(p, yb).item() * len(xb)
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    n = len(all_labels)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    return (
        total_loss / n,
        float(accuracy_score(all_labels, preds)),
        float(roc_auc_score(all_labels, all_probs)),
    )


def mc_score(model: nn.Module, X: np.ndarray,
             device: torch.device, T: int = T_MC) -> np.ndarray:
    """MC-Dropout mean probs."""
    model.train()  # enable dropout
    t = torch.tensor(X).unsqueeze(1).to(device)
    passes = []
    with torch.no_grad():
        for _ in range(T):
            passes.append(model(t).cpu().numpy())
    return np.stack(passes).mean(axis=0)


# ---------------------------------------------------------------------------
# Knowledge lookup (static — not seed-dependent)
# ---------------------------------------------------------------------------
def build_knowledge_lookup():
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

    def lookup(cn: str) -> np.ndarray:
        if cn in family_mean:
            return family_mean[cn]
        for k in family_mean:
            if k.lower() in cn.lower() or cn.lower() in k.lower():
                return family_mean[k]
        return zero_vec

    return lookup, zero_vec, feat_cols


def score_knowledge_batch(X_know: np.ndarray) -> np.ndarray:
    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    agent = KnowledgeAgent(
        checkpoint_path=str(CKPT_DIR / "knowledge_cnn.pt"),
        metadata_path=str(CKPT_DIR / "metadata.json"),
        T=T_MC,
    )
    outs = agent.score_batch([{"features": X_know[i]} for i in range(len(X_know))])
    return np.array([o.prob for o in outs])


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def train_one(seed: int, split: str, df: pd.DataFrame,
              device: torch.device, ckpt_dir: Path) -> dict:
    set_seed(seed)
    d_tr, d_vl, d_te = load_split_data(df, split)

    X_tr = d_tr[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = d_tr["label"].astype(np.float32).values
    X_vl = d_vl[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_vl = d_vl["label"].astype(np.float32).values
    X_te = d_te[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_te = d_te["label"].astype(np.float32).values
    fam_te = d_te["class_name"].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_vl = scaler.transform(X_vl).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    sm = SMOTE(random_state=seed)
    X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)

    train_ds = TensorDataset(torch.tensor(X_tr_r).unsqueeze(1), torch.tensor(y_tr_r))
    val_ds   = TensorDataset(torch.tensor(X_vl).unsqueeze(1),   torch.tensor(y_vl))
    test_ds  = TensorDataset(torch.tensor(X_te).unsqueeze(1),   torch.tensor(y_te))
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = CNNBehavioralAgent().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.BCELoss()

    best_val_auc   = -1.0
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        _, _, val_auc = evaluate_model(model, val_ld, device)
        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    _, _, test_auc = evaluate_model(model, test_ld, device)

    # AUPRC requires collecting all probs
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            p = model(xb.to(device)).cpu().numpy()
            all_probs.extend(p)
            all_labels.extend(yb.numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)
    test_auprc = float(average_precision_score(all_labels, all_probs))
    test_acc   = float(accuracy_score(all_labels, preds))

    # Save seed-specific checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"seed{seed}_{split}.pt")
    joblib.dump(scaler, ckpt_dir / f"seed{seed}_{split}_scaler.joblib")

    result = {
        "seed": seed, "split": split,
        "best_val_auc": round(float(best_val_auc), 4),
        "test_auroc":   round(float(test_auc),    4),
        "test_auprc":   round(float(test_auprc),  4),
        "test_acc":     round(float(test_acc),     4),
    }

    # [B+K] fusion for random_stratified only
    if split == "random_stratified":
        kn_lookup, zero_vec, _ = build_knowledge_lookup()
        X_know = np.stack([kn_lookup(f) for f in fam_te])
        b_probs = mc_score(model, X_te, device)
        k_probs = score_knowledge_batch(X_know)
        bk_probs = (b_probs + k_probs) / 2.0
        result["bk_test_auroc"] = round(float(roc_auc_score(all_labels, bk_probs)), 4)
        result["bk_test_auprc"] = round(float(average_precision_score(all_labels, bk_probs)), 4)

    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Returns (mean, std, ci_lo, ci_hi) using t-distribution."""
    arr = np.array(values)
    n   = len(arr)
    m   = float(arr.mean())
    s   = float(arr.std(ddof=1))
    tc  = float(t_dist.ppf(0.975, df=n - 1))
    hw  = tc * s / np.sqrt(n)
    return m, s, m - hw, m + hw


def print_summary(all_results: list[dict]) -> dict:
    summary: dict = {}

    for split in SPLITS:
        subset = [r for r in all_results if r["split"] == split]
        if not subset:
            continue

        aurocs = [r["test_auroc"] for r in subset]
        auprcs = [r["test_auprc"] for r in subset]
        accs   = [r["test_acc"]   for r in subset]

        m_a, s_a, lo_a, hi_a = ci95(aurocs)
        m_p, s_p, lo_p, hi_p = ci95(auprcs)
        m_c, s_c, lo_c, hi_c = ci95(accs)

        print(f"\n  {split}")
        print(f"    AUROC : {m_a:.4f} ± {s_a:.4f}  95% CI [{lo_a:.4f}, {hi_a:.4f}]")
        print(f"    AUPRC : {m_p:.4f} ± {s_p:.4f}  95% CI [{lo_p:.4f}, {hi_p:.4f}]")
        print(f"    Acc   : {m_c:.4f} ± {s_c:.4f}  95% CI [{lo_c:.4f}, {hi_c:.4f}]")
        print(f"    Seeds : {[r['seed'] for r in subset]}")
        print(f"    Values: {aurocs}")

        entry: dict = {
            "n_seeds": len(subset),
            "seeds":   [r["seed"] for r in subset],
            "auroc":   {"mean": round(m_a,4), "std": round(s_a,4),
                        "ci95_lo": round(lo_a,4), "ci95_hi": round(hi_a,4),
                        "values": aurocs},
            "auprc":   {"mean": round(m_p,4), "std": round(s_p,4),
                        "ci95_lo": round(lo_p,4), "ci95_hi": round(hi_p,4),
                        "values": auprcs},
            "acc":     {"mean": round(m_c,4), "std": round(s_c,4),
                        "ci95_lo": round(lo_c,4), "ci95_hi": round(hi_c,4),
                        "values": accs},
        }

        # [B+K] stats for random_stratified
        if split == "random_stratified":
            bk_aurocs = [r["bk_test_auroc"] for r in subset if "bk_test_auroc" in r]
            bk_auprcs = [r["bk_test_auprc"] for r in subset if "bk_test_auprc" in r]
            if bk_aurocs:
                bm_a, bs_a, blo_a, bhi_a = ci95(bk_aurocs)
                bm_p, bs_p, blo_p, bhi_p = ci95(bk_auprcs)
                print(f"    [B+K] AUROC: {bm_a:.4f} ± {bs_a:.4f}  "
                      f"95% CI [{blo_a:.4f}, {bhi_a:.4f}]")
                print(f"    [B+K] AUPRC: {bm_p:.4f} ± {bs_p:.4f}  "
                      f"95% CI [{blo_p:.4f}, {bhi_p:.4f}]")
                entry["bk_auroc"] = {"mean": round(bm_a,4), "std": round(bs_a,4),
                                     "ci95_lo": round(blo_a,4), "ci95_hi": round(bhi_a,4),
                                     "values": bk_aurocs}
                entry["bk_auprc"] = {"mean": round(bm_p,4), "std": round(bs_p,4),
                                     "ci95_lo": round(blo_p,4), "ci95_hi": round(bhi_p,4),
                                     "values": bk_auprcs}

        summary[split] = entry

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Splits: {SPLITS}")
    print(f"Seeds : {SEEDS} + canonical seed 42 from metadata")
    print(f"Total training runs: {len(SPLITS) * len(SEEDS)}")

    print("\nLoading dataset ...")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} windows")

    all_results: list[dict] = []

    # -----------------------------------------------------------------------
    # Load seed-42 results from metadata
    # -----------------------------------------------------------------------
    meta = json.loads((CKPT_DIR / "metadata.json").read_text())
    for split in SPLITS:
        key = f"behavioral_{split}"
        if key in meta:
            m = meta[key]
            r42: dict = {
                "seed": 42, "split": split,
                "best_val_auc": m.get("best_val_auc"),
                "test_auroc":   m.get("test_auc"),
                "test_auprc":   None,   # metadata doesn't store AUPRC — compute below
                "test_acc":     m.get("test_accuracy"),
            }
            all_results.append(r42)
            print(f"  Loaded seed-42 {split}: AUROC={r42['test_auroc']}")

    # Re-compute seed-42 AUPRC and [B+K] from existing checkpoints
    print("\nComputing missing seed-42 metrics (AUPRC, [B+K]) ...")
    kn_lookup, zero_vec, _ = build_knowledge_lookup()
    for split in SPLITS:
        r42 = next(r for r in all_results if r["seed"] == 42 and r["split"] == split)
        ckpt = CKPT_DIR / f"behavioral_cnn_{split}.pt"
        scl  = CKPT_DIR / f"behavioral_scaler_{split}.joblib"
        if not ckpt.exists():
            continue

        model = CNNBehavioralAgent().to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        scaler = joblib.load(str(scl))

        # Load test data
        _, _, d_te = load_split_data(df, split)
        X_te   = scaler.transform(
                     d_te[FEATURE_COLS].to_numpy(dtype=np.float32)
                 ).astype(np.float32)
        y_te   = d_te["label"].astype(np.float32).values
        fam_te = d_te["class_name"].values

        t_ten = torch.tensor(X_te).unsqueeze(1).to(device)
        with torch.no_grad():
            probs42 = model(t_ten).cpu().numpy()

        r42["test_auprc"] = round(float(average_precision_score(y_te, probs42)), 4)

        if split == "random_stratified":
            X_know = np.stack([kn_lookup(f) for f in fam_te])
            k_probs = score_knowledge_batch(X_know)
            bk_probs = (probs42 + k_probs) / 2.0
            r42["bk_test_auroc"] = round(float(roc_auc_score(y_te, bk_probs)), 4)
            r42["bk_test_auprc"] = round(float(average_precision_score(y_te, bk_probs)), 4)

        print(f"  seed-42 {split}: AUPRC={r42['test_auprc']}")

    # -----------------------------------------------------------------------
    # Train seeds 0-4
    # -----------------------------------------------------------------------
    total = len(SPLITS) * len(SEEDS)
    done  = 0
    for split in SPLITS:
        for seed in SEEDS:
            done += 1
            print(f"\n[{done:2d}/{total}] seed={seed}  split={split}")
            r = train_one(seed, split, df, device, SEED_DIR)
            all_results.append(r)
            print(f"         AUROC={r['test_auroc']}  AUPRC={r['test_auprc']}  "
                  f"Acc={r['test_acc']}", end="")
            if "bk_test_auroc" in r:
                print(f"  [B+K] AUROC={r['bk_test_auroc']}", end="")
            print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  MULTI-SEED SUMMARY  (6 seeds: 0,1,2,3,4,42)")
    print(f"{'='*65}")
    summary = print_summary(all_results)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "seeds":      SEEDS + [42],
        "splits":     SPLITS,
        "per_run":    all_results,
        "summary":    summary,
    }
    out_path = RESULTS / "multiseed.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
