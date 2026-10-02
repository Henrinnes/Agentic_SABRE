#!/usr/bin/env python3
"""One-off: genuine storage-only / memory-only / combined-16D Behavioural
feature-subset ablation for fig_feature_ablation / tab:feature_ablation.

No existing script performs this specific ablation (results/ablation_e5.json
is a different, modality-combination ablation). Trains 3 feature-subset
CNNBehavioralAgent variants x K=6 seeds each on random_stratified (same
recipe as retrain_behavioural.py: SMOTE on train, early-stop on val AUC,
scaler fit on train only), reports test AUROC/AUPRC mean +/- 95% CI.

Saves results/feature_ablation.json.
"""
import os
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import t as scipy_t

PARQUET    = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "ransmap_extracted/ransmap_train.parquet")
SPLITS_DIR = Path("splits")
RESULTS    = Path("results")
SPLIT      = "random_stratified"

EPOCHS   = 30
BATCH    = 256
LR       = 1e-3
DROPOUT  = 0.3
PATIENCE = 7
K_SEEDS  = 6

STORAGE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean",
]
MEMORY_COLS = [
    "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]
COMBINED_COLS = STORAGE_COLS + MEMORY_COLS
assert len(STORAGE_COLS) == 5 and len(MEMORY_COLS) == 11 and len(COMBINED_COLS) == 16

VARIANTS = {
    "storage_only": STORAGE_COLS,
    "memory_only":  MEMORY_COLS,
    "combined_16d": COMBINED_COLS,
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        pooled = input_dim // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout), nn.Flatten(),
            nn.Linear(pooled * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)


def load_sessions(subset):
    return set((SPLITS_DIR / SPLIT / f"{subset}_sessions.txt").read_text().splitlines())


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.extend(model(xb.to(device)).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(probs), np.array(labels)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

df = pd.read_parquet(PARQUET)
train_sess, val_sess, test_sess = (load_sessions(s) for s in ("train", "val", "test"))
df_train = df[df["session"].isin(train_sess)].reset_index(drop=True)
df_val   = df[df["session"].isin(val_sess)].reset_index(drop=True)
df_test  = df[df["session"].isin(test_sess)].reset_index(drop=True)
print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

y_train = df_train["label"].astype(np.float32).values
y_val   = df_val["label"].astype(np.float32).values
y_test  = df_test["label"].astype(np.float32).values

results = {"k_seeds": K_SEEDS, "split": SPLIT, "variants": {}}

for vname, cols in VARIANTS.items():
    print(f"\n{'='*60}\nVariant: {vname}  ({len(cols)}D)\n{'='*60}")
    aurocs, auprcs = [], []
    for seed in range(K_SEEDS):
        set_seed(seed)
        X_train = df_train[cols].to_numpy(dtype=np.float32)
        X_val   = df_val[cols].to_numpy(dtype=np.float32)
        X_test  = df_test[cols].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val   = scaler.transform(X_val).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        sm = SMOTE(random_state=seed)
        X_train_r, y_train_r = sm.fit_resample(X_train, y_train)

        train_ds = TensorDataset(torch.tensor(X_train_r).unsqueeze(1), torch.tensor(y_train_r))
        val_ds   = TensorDataset(torch.tensor(X_val).unsqueeze(1),   torch.tensor(y_val))
        test_ds  = TensorDataset(torch.tensor(X_test).unsqueeze(1),  torch.tensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

        model = CNNBehavioralAgent(len(cols), DROPOUT).to(device)
        opt  = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.BCELoss()

        best_val_auc, patience_count, best_state = -1.0, 0, None
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            val_probs, val_labels = evaluate(model, val_loader, device)
            val_auc = roc_auc_score(val_labels, val_probs)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        test_probs, test_labels = evaluate(model, test_loader, device)
        auroc = roc_auc_score(test_labels, test_probs)
        auprc = average_precision_score(test_labels, test_probs)
        aurocs.append(auroc); auprcs.append(auprc)
        print(f"  seed={seed}  best_val_auc={best_val_auc:.4f}  test_auroc={auroc:.4f}  test_auprc={auprc:.4f}")

    def ci95(vals):
        a = np.array(vals); n = len(a); m = float(a.mean())
        se = float(a.std(ddof=1)) / np.sqrt(n)
        lo, hi = scipy_t.interval(0.95, df=n-1, loc=m, scale=se)
        return {"mean": round(m, 4), "ci95_lo": round(float(lo), 4),
                "ci95_hi": round(float(hi), 4), "half_width": round((hi-lo)/2, 4)}

    results["variants"][vname] = {
        "n_features": len(cols), "features": cols,
        "auroc": ci95(aurocs), "auprc": ci95(auprcs),
        "auroc_seeds": [round(a, 4) for a in aurocs],
        "auprc_seeds": [round(a, 4) for a in auprcs],
    }
    print(f"  {vname}: AUROC={results['variants'][vname]['auroc']}")

out_path = RESULTS / "feature_ablation.json"
out_path.write_text(json.dumps(results, indent=2))
print("\nsaved ->", out_path)
