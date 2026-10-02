#!/usr/bin/env python3
"""
run_behavioural_baselines.py — R1#11: tabular baselines + deep-ensemble
uncertainty comparison for the Behavioural Agent.

Reviewer 1 asked for the Behavioural CNN to be compared against strong
tabular baselines on the identical 16D feature set (since a CNN over 16
manually-ordered heterogeneous features needs justification -- convolution
assumes local feature relationships a tree/linear model does not), and for
at least one deep-ensemble uncertainty baseline against MC Dropout.

E-H  Tabular baselines: LogisticRegression, RandomForest, GradientBoosting
     (XGBoost/LightGBM substitute -- neither is installed in this
     environment; sklearn's GBT is already used elsewhere in this repo for
     the same reason), MLPClassifier (ordinary MLP, no MC Dropout), SVC
     (RBF kernel). All trained on the identical SMOTE-balanced
     random-stratified 16D training split used for the Behavioural CNN,
     evaluated on the same test split.

E-I  Deep ensemble: the 5 independently-retrained Behavioural CNN seeds in
     checkpoints/seeds/ (seed0-4) plus the canonical seed-42 checkpoint form
     a genuine 6-member deep ensemble (Lakshminarayanan et al. 2017).
     Ensemble mean/std (single deterministic forward pass per member) is
     compared against MC Dropout (T=10 stochastic passes, single
     checkpoint) on AUROC, calibration, and uncertainty-error correlation.

Outputs: results/behavioural_baselines.json
"""
from __future__ import annotations

import os

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT    = Path(__file__).parent
CKPT    = ROOT / "checkpoints"
SPLITS  = ROOT / "splits"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RANSMAP = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "ransmap_extracted/ransmap_train.parquet")

BEHAVIOR_COLS = [
    "avg_read_throughput","avg_write_throughput",
    "read_LBA_variance","write_LBA_variance",
    "write_entropy_mean","mem_read_entropy_mean","mem_write_entropy_mean",
    "mem_readwrite_entropy_mean","mem_read_GPA_variance",
    "mem_write_GPA_variance","mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance","mem_read_count","mem_write_count",
    "mem_readwrite_count","mem_exec_count",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_MC   = 10
SEED   = 42

class CNNBehavioral(nn.Module):
    def __init__(self, in_dim=16, p=0.3):
        super().__init__()
        pooled = (in_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1,16,3,padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
            nn.Linear(pooled*16,64), nn.ReLU(),
            nn.Linear(64,1), nn.Sigmoid(),
        )
    def forward(self,x): return self.model(x).squeeze(-1)

def _enable_mc(m):
    if isinstance(m, nn.Dropout): m.train()

def load_split(split: str, subset: str):
    sess = set((SPLITS/split/f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[BEHAVIOR_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    return X, y

def load_cnn(path_stub: str):
    m = CNNBehavioral(16)
    m.load_state_dict(torch.load(str(CKPT/f"{path_stub}.pt"), map_location="cpu"))
    m.eval().to(DEVICE)
    # Canonical checkpoints use behavioral_scaler_{split}.joblib; per-seed
    # checkpoints (checkpoints/seeds/) use {stub}_scaler.joblib. Try both.
    suffix_path = CKPT / f"{path_stub}_scaler.joblib"
    if suffix_path.exists():
        sc = joblib.load(str(suffix_path))
    else:
        split_name = path_stub.split("/")[-1].replace("behavioral_cnn_", "")
        sc = joblib.load(str(CKPT / f"behavioral_scaler_{split_name}.joblib"))
    return m, sc

def mc_infer(model, X, T=T_MC):
    model.apply(_enable_mc)
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    preds = []
    with torch.no_grad():
        for _ in range(T): preds.append(model(t).cpu().numpy())
    preds = np.stack(preds)
    return preds.mean(0), preds.std(0)

def det_infer(model, X):
    model.eval()
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        return model(t).cpu().numpy()

def ece_score(y_true, y_prob, n_bins=15):
    try:
        fracs, means = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        weights = counts / (counts.sum() + 1e-9)
        return float(np.sum(weights[:len(fracs)] * np.abs(fracs - means)))
    except Exception:
        return float("nan")

print("Loading random_stratified 16D Behavioural features …")
X_tr, y_tr = load_split("random_stratified", "train")
X_te, y_te = load_split("random_stratified", "test")
print(f"  train={len(y_tr)} (ransom={y_tr.mean():.3f})  test={len(y_te)} (ransom={y_te.mean():.3f})")

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr).astype(np.float32)
X_te_s = sc.transform(X_te).astype(np.float32)

print("Applying SMOTE to training split (matches Behavioural CNN preprocessing) …")
X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr_s, y_tr)
print(f"  post-SMOTE train={len(y_tr_sm)} (ransom={y_tr_sm.mean():.3f})")

results: dict = {}

# ═════════════════════════════════════════════════════════════════════════════
# E-H  Tabular baselines
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-H  Tabular Baselines (identical 16D feature set)")
print("="*65)

baselines = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "random_forest":       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "gradient_boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=SEED),
    "mlp_ordinary":        MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=SEED, early_stopping=True),
    "svm_rbf":             SVC(kernel="rbf", probability=True, random_state=SEED, cache_size=1000),
}

eh_results = {}
for name, clf in baselines.items():
    t0 = time.time()
    print(f"  Training {name} …", end=" ", flush=True)
    clf.fit(X_tr_sm, y_tr_sm)
    probs = clf.predict_proba(X_te_s)[:, 1]
    auroc = roc_auc_score(y_te, probs)
    auprc = average_precision_score(y_te, probs)
    ece   = ece_score(y_te, probs)
    dt = time.time() - t0
    eh_results[name] = {"auroc": round(float(auroc), 4),
                         "auprc": round(float(auprc), 4),
                         "ece":   round(float(ece), 4),
                         "train_seconds": round(dt, 1)}
    print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ECE={ece:.4f}  ({dt:.1f}s)")

# Behavioural CNN reference (canonical checkpoint, MC Dropout mean)
cnn_m, cnn_sc = load_cnn("behavioral_cnn_random_stratified")
X_te_cnn = cnn_sc.transform(X_te).astype(np.float32)
p_cnn, sig_cnn = mc_infer(cnn_m, X_te_cnn)
auroc_cnn = roc_auc_score(y_te, p_cnn)
auprc_cnn = average_precision_score(y_te, p_cnn)
ece_cnn   = ece_score(y_te, p_cnn)
eh_results["behavioural_cnn_mc_dropout"] = {
    "auroc": round(float(auroc_cnn), 4), "auprc": round(float(auprc_cnn), 4),
    "ece": round(float(ece_cnn), 4),
}
print(f"  {'behavioural_cnn (reference)':22s}  AUROC={auroc_cnn:.4f}  AUPRC={auprc_cnn:.4f}  ECE={ece_cnn:.4f}")
results["E_H"] = eh_results

# ═════════════════════════════════════════════════════════════════════════════
# E-I  Deep ensemble vs MC Dropout
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-I  Deep Ensemble (6 independently-trained seeds) vs MC Dropout")
print("="*65)

seed_stubs = [f"seeds/seed{k}_random_stratified" for k in range(5)] + ["behavioral_cnn_random_stratified"]
ensemble_preds = []
for stub in seed_stubs:
    m, s = load_cnn(stub)
    Xs = s.transform(X_te).astype(np.float32)
    ensemble_preds.append(det_infer(m, Xs))
ensemble_preds = np.stack(ensemble_preds)  # (6, n_test)
p_ens   = ensemble_preds.mean(0)
sig_ens = ensemble_preds.std(0)

auroc_ens = roc_auc_score(y_te, p_ens)
auprc_ens = average_precision_score(y_te, p_ens)
ece_ens   = ece_score(y_te, p_ens)

# Uncertainty-error correlation: do higher-uncertainty samples have higher
# error rate? Bin by uncertainty decile, report error rate per decile.
err_mc  = np.abs((p_cnn >= 0.5).astype(int) - y_te)
err_ens = np.abs((p_ens >= 0.5).astype(int) - y_te)

def decile_error_curve(sig, err, n_bins=10):
    order = np.argsort(sig)
    bins = np.array_split(order, n_bins)
    return [float(err[b].mean()) for b in bins]

mc_curve  = decile_error_curve(sig_cnn, err_mc)
ens_curve = decile_error_curve(sig_ens, err_ens)

results["E_I"] = {
    "mc_dropout": {
        "auroc": round(float(auroc_cnn), 4), "auprc": round(float(auprc_cnn), 4),
        "ece": round(float(ece_cnn), 4),
        "error_rate_by_uncertainty_decile": [round(v, 4) for v in mc_curve],
    },
    "deep_ensemble": {
        "n_members": len(seed_stubs),
        "auroc": round(float(auroc_ens), 4), "auprc": round(float(auprc_ens), 4),
        "ece": round(float(ece_ens), 4),
        "error_rate_by_uncertainty_decile": [round(v, 4) for v in ens_curve],
    },
}
print(f"  MC Dropout    (T={T_MC}, 1 ckpt):  AUROC={auroc_cnn:.4f}  AUPRC={auprc_cnn:.4f}  ECE={ece_cnn:.4f}")
print(f"  Deep Ensemble (6 ckpts):          AUROC={auroc_ens:.4f}  AUPRC={auprc_ens:.4f}  ECE={ece_ens:.4f}")
print(f"  MC Dropout error rate by uncertainty decile (low->high):  {mc_curve}")
print(f"  Deep Ensemble error rate by uncertainty decile (low->high): {ens_curve}")

out_path = RESULTS / "behavioural_baselines.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out_path}")
print("Done.")
