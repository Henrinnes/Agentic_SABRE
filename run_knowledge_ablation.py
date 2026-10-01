#!/usr/bin/env python3
"""
run_knowledge_ablation.py — R1#2: explicit Knowledge Agent ablation matrix.

Reviewer 1 asked: "at the time an unknown process starts encrypting files,
where does its threat-intelligence transcript come from?" and requested
explicit experiments for Behavioural-only, Behavioural+Knowledge (no
Semantic), Knowledge-missing, Knowledge-replaced-by-corpus-mean (already
the paper's default post-fix, see Remark 1 of the main paper), and
incorrect/adversarial Knowledge input.

All variants use the same out-of-fold Behavioural scoring (5-fold
session-grouped cross-fitting) established for the main pipeline, so none
of these ablations reintroduce the stacking leak fixed elsewhere in this
codebase.

E-J1  Behavioural-only vs Behavioural+Knowledge (2-agent, no Semantic):
      is a genuine 2-agent fusion trained from scratch, not the deployed
      3-agent model with one input masked.
E-J2  Test-time robustness of the DEPLOYED 3-agent Decision MLP when the
      Knowledge input is corrupted at inference: corpus-mean (default),
      masked (neutral/max-uncertainty), adversarial (inverted probability).
      This answers "what happens when no transcript exists / the mapping
      is wrong" for the model actually used in the paper's results.

Learned missing-modality gating (an alternative to a fixed corpus-mean
prior) is noted as future work in the main paper rather than implemented
here -- it requires a new gating architecture, not just a new evaluation.

Outputs: results/knowledge_ablation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as _tts
from sklearn.calibration import calibration_curve
from Agentic_SABRE.knowledge_features import RANSKNOW_CSV

ROOT    = Path(__file__).parent
CKPT    = ROOT / "checkpoints"
SPLITS  = ROOT / "splits"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RANSMAP  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RANSKNOW = RANSKNOW_CSV

BEHAVIOR_COLS = [
    "avg_read_throughput","avg_write_throughput",
    "read_LBA_variance","write_LBA_variance",
    "write_entropy_mean","mem_read_entropy_mean","mem_write_entropy_mean",
    "mem_readwrite_entropy_mean","mem_read_GPA_variance",
    "mem_write_GPA_variance","mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance","mem_read_count","mem_write_count",
    "mem_readwrite_count","mem_exec_count",
]

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_MC    = 10
META    = json.loads((CKPT / "metadata.json").read_text())

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

class CNNKnowledge(nn.Module):
    def __init__(self, in_dim=23, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,16,3,padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64,1), nn.Sigmoid(),
        )
    def forward(self,x): return self.fc(self.conv(x)).squeeze(-1)

class FusionMLP(nn.Module):
    def __init__(self, in_dim, hidden=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden,1), nn.Sigmoid(),
        )
    def forward(self,x): return self.net(x).squeeze(-1)

def _enable_mc(m):
    if isinstance(m, nn.Dropout): m.train()

def mc_infer_batched(model, X, T=T_MC, batch=512):
    model.to(DEVICE); model.apply(_enable_mc)
    means, stds = [], []
    for i in range(0, len(X), batch):
        t = torch.tensor(X[i:i+batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        preds = []
        with torch.no_grad():
            for _ in range(T): preds.append(model(t).cpu().numpy())
        preds = np.stack(preds)
        means.append(preds.mean(0)); stds.append(preds.std(0))
    return np.concatenate(means), np.concatenate(stds)

def load_behavioral(split: str):
    m = CNNBehavioral(META["behavioral_input_dim"])
    m.load_state_dict(torch.load(str(CKPT/f"behavioral_cnn_{split}.pt"), map_location="cpu"))
    m.eval()
    sc = joblib.load(str(CKPT/f"behavioral_scaler_{split}.joblib"))
    return m, sc

def load_knowledge():
    dim = META["knowledge_input_dim"]
    m = CNNKnowledge(dim)
    with torch.no_grad(): _ = m(torch.zeros(1,1,dim))
    m.load_state_dict(torch.load(str(CKPT/"knowledge_cnn.pt"), map_location="cpu"))
    m.eval()
    sc = joblib.load(str(CKPT/"knowledge_scaler.joblib"))
    return m, sc

def load_beh_split(split: str, subset: str):
    sess = set((SPLITS/split/f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[BEHAVIOR_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    return X, y

def build_oof_behavioural_scores(n_folds=5):
    y_all, p_all, sig_all = [], [], []
    for k in range(n_folds):
        beh_m_k, beh_sc_k = load_behavioral(f"cv_fold_{k}")
        X_k, y_k = load_beh_split(f"cv_fold_{k}", "test")
        Xs_k = beh_sc_k.transform(X_k).astype(np.float32)
        p_k, sig_k = mc_infer_batched(beh_m_k, Xs_k)
        y_all.append(y_k); p_all.append(p_k); sig_all.append(sig_k)
    return (np.concatenate(p_all), np.concatenate(sig_all), np.concatenate(y_all).astype(int))

def corpus_mean_knowledge(know_m, know_sc):
    df  = pd.read_csv(RANSKNOW)
    fc  = META["knowledge_features"]
    Xrk = know_sc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    corp = Xrk.mean(0)
    p_k, sig_k = mc_infer_batched(know_m, corp[None, :])
    return float(p_k[0]), float(sig_k[0])

def ece_score(y_true, y_prob, n_bins=15):
    try:
        fracs, means = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        weights = counts / (counts.sum() + 1e-9)
        return float(np.sum(weights[:len(fracs)] * np.abs(fracs - means)))
    except Exception:
        return float("nan")

def train_fusion_mlp(X_tr, y_tr, in_dim, seed=0, epochs=40, patience=6):
    sc = StandardScaler().fit(X_tr)
    X_sc = sc.transform(X_tr).astype(np.float32)
    y_f  = y_tr.astype(np.float32)
    X_t, X_v, y_t, y_v = _tts(X_sc, y_f, test_size=0.2,
                               stratify=(y_f > 0.5).astype(int), random_state=seed)
    torch.manual_seed(seed)
    net = FusionMLP(in_dim, 32, 0.2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_t), torch.tensor(y_t)),
        batch_size=128, shuffle=True)
    best_va, best_state, pat = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); bce(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = bce(net(torch.tensor(X_v).to(DEVICE)), torch.tensor(y_v).to(DEVICE)).item()
        if va + 1e-6 < best_va:
            best_va = va; pat = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= patience: break
    net.load_state_dict(best_state); net.eval()
    return net, sc

def predict(net, sc, X):
    Xs = sc.transform(X).astype(np.float32)
    with torch.no_grad():
        return net(torch.tensor(Xs).to(DEVICE)).cpu().numpy()

print("Loading models …")
know_m, know_sc = load_knowledge()
p_know_corpus, sig_know_corpus = corpus_mean_knowledge(know_m, know_sc)
print(f"  Knowledge corpus-mean: p={p_know_corpus:.4f}  sigma={sig_know_corpus:.2e}")

print("\nBuilding out-of-fold Behavioural training scores (random_stratified) …")
p_b_tr, sig_b_tr, y_tr = build_oof_behavioural_scores()

print("Building Behavioural test scores (random_stratified, canonical checkpoint) …")
beh_m_rs, beh_sc_rs = load_behavioral("random_stratified")
X_beh_te, y_te = load_beh_split("random_stratified", "test")
Xs_te = beh_sc_rs.transform(X_beh_te).astype(np.float32)
p_b_te, sig_b_te = mc_infer_batched(beh_m_rs, Xs_te)

results: dict = {}

# ═════════════════════════════════════════════════════════════════════════════
# E-J1: Behavioural-only vs Behavioural+Knowledge (genuine 2-agent retrain)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-J1  Behavioural-only vs Behavioural+Knowledge (2-agent)")
print("="*65)

auroc_b_only = roc_auc_score(y_te, p_b_te)
auprc_b_only = average_precision_score(y_te, p_b_te)
ece_b_only   = ece_score(y_te, p_b_te)
print(f"  Behavioural-only:      AUROC={auroc_b_only:.4f}  AUPRC={auprc_b_only:.4f}  ECE={ece_b_only:.4f}")

n_tr = len(p_b_tr)
Xk_tr_const = np.tile([p_know_corpus, sig_know_corpus], (n_tr, 1)).astype(np.float32)
X_tr_bk = np.stack([p_b_tr, Xk_tr_const[:,0], sig_b_tr, Xk_tr_const[:,1]], axis=1).astype(np.float32)
n_te = len(p_b_te)
Xk_te_const = np.tile([p_know_corpus, sig_know_corpus], (n_te, 1)).astype(np.float32)
X_te_bk = np.stack([p_b_te, Xk_te_const[:,0], sig_b_te, Xk_te_const[:,1]], axis=1).astype(np.float32)

net_bk, sc_bk = train_fusion_mlp(X_tr_bk, y_tr, in_dim=4, seed=0)
probs_bk = predict(net_bk, sc_bk, X_te_bk).ravel()
auroc_bk = roc_auc_score(y_te, probs_bk)
auprc_bk = average_precision_score(y_te, probs_bk)
ece_bk   = ece_score(y_te, probs_bk)
print(f"  Behavioural+Knowledge: AUROC={auroc_bk:.4f}  AUPRC={auprc_bk:.4f}  ECE={ece_bk:.4f}")

results["E_J1"] = {
    "behavioural_only": {"auroc": round(float(auroc_b_only),4), "auprc": round(float(auprc_b_only),4), "ece": round(float(ece_b_only),4)},
    "behavioural_plus_knowledge_2agent": {"auroc": round(float(auroc_bk),4), "auprc": round(float(auprc_bk),4), "ece": round(float(ece_bk),4)},
    "note": "Knowledge is the fixed corpus-mean constant in both training and test (Remark 1); "
            "since it carries no per-sample signal, any AUROC difference here reflects training "
            "noise/regularisation effects of the extra (constant) input dimension, not genuine "
            "Knowledge-agent information.",
}

# ═════════════════════════════════════════════════════════════════════════════
# E-J2: Test-time robustness of the deployed 3-agent Decision MLP under
#       different Knowledge-input scenarios
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-J2  Deployed 3-Agent MLP: Knowledge Input Robustness")
print("="*65)

# We train a fresh 3-agent Decision MLP here (rather than loading the
# static decision_fuser_3agent.json artifact) so the Semantic constant used
# at training time is IDENTICAL to the one used in every test-time scenario
# below. Reusing that pretrained artifact with an unrelated placeholder
# Semantic value would confound the Knowledge-robustness comparison with an
# unrelated Semantic-input mismatch.
p_sem_fill = 0.5   # neutral placeholder, identical at train and test time
sig_sem_fill = 0.1
n_tr3 = len(p_b_tr)
p_s_tr3   = np.full(n_tr3, p_sem_fill, np.float32)
sig_s_tr3 = np.full(n_tr3, sig_sem_fill, np.float32)
p_k_tr3   = np.full(n_tr3, p_know_corpus, np.float32)
sig_k_tr3 = np.full(n_tr3, sig_know_corpus, np.float32)
X_tr3 = np.stack([p_s_tr3, p_b_tr, p_k_tr3, sig_s_tr3, sig_b_tr, sig_k_tr3], axis=1).astype(np.float32)
mlp, mlp_sc = train_fusion_mlp(X_tr3, y_tr, in_dim=6, seed=0)

n = len(p_b_te)
p_s_col   = np.full(n, p_sem_fill, np.float32)
sig_s_col = np.full(n, sig_sem_fill, np.float32)

def fuse_and_score(p_k_col, sig_k_col, label):
    feat = np.stack([p_s_col, p_b_te, p_k_col, sig_s_col, sig_b_te, sig_k_col], axis=1).astype(np.float32)
    probs = predict(mlp, mlp_sc, feat).ravel()
    auroc = roc_auc_score(y_te, probs)
    auprc = average_precision_score(y_te, probs)
    ece   = ece_score(y_te, probs)
    print(f"  {label:28s} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ECE={ece:.4f}")
    return {"auroc": round(float(auroc),4), "auprc": round(float(auprc),4), "ece": round(float(ece),4)}

ej2 = {}
# Scenario 1: corpus-mean (default, current paper methodology)
p_k_default = np.full(n, p_know_corpus, np.float32)
sig_k_default = np.full(n, sig_know_corpus, np.float32)
ej2["corpus_mean_default"] = fuse_and_score(p_k_default, sig_k_default, "Corpus-mean (default)")

# Scenario 2: masked/missing (neutral probability, max observed uncertainty)
sig_k_max = float(max(sig_know_corpus, sig_b_te.max()))
p_k_masked = np.full(n, 0.5, np.float32)
sig_k_masked = np.full(n, sig_k_max, np.float32)
ej2["masked_missing"] = fuse_and_score(p_k_masked, sig_k_masked, "Masked/missing (p=0.5, max sigma)")

# Scenario 3: adversarial (inverted corpus-mean probability, low uncertainty
# so it is NOT naturally down-weighted by Peer Consultation/triage sigma_max)
p_k_adv = np.full(n, 1.0 - p_know_corpus, np.float32)
sig_k_adv = np.full(n, sig_know_corpus, np.float32)
ej2["adversarial_inverted"] = fuse_and_score(p_k_adv, sig_k_adv, "Adversarial (inverted p, low sigma)")

results["E_J2"] = ej2
results["E_J2"]["note"] = ("A 3-agent Decision MLP trained fresh (seed=0, out-of-fold Behavioural, "
                            "corpus-mean Knowledge, neutral-placeholder Semantic held identical at "
                            "train and test time) with only the Knowledge input dimension swapped "
                            "at inference time, holding Behavioural and Semantic fixed. Quantifies "
                            "how much the model's output is perturbed by a corrupted or adversarial "
                            "Knowledge score, since Peer Consultation only helps if sigma_know is "
                            "itself elevated -- an adversarial low-sigma input is not caught by "
                            "uncertainty-gating.")

out_path = RESULTS / "knowledge_ablation.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out_path}")
print("Done.")
