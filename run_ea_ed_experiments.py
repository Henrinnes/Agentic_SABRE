#!/usr/bin/env python3
"""
run_ea_ed_experiments.py
========================
Implements four post-hoc experiments (E-A through E-D) using existing
checkpoints and results — no model retraining required.

  E-A  Statistical significance   Paired Wilcoxon (B vs MLP, each split) +
                                   McNemar per-sample disagreement test
  E-B  Risk-coverage (AURC)       Selective-prediction curve: precision vs
                                   auto-classification coverage for the MLP
  E-C  Conformal coverage shift    Empirical Mondrian CP coverage on all four
                                   splits (calibrate on val, evaluate on test)
  E-D  Session-level aggregation  Max-score and majority-vote per RanSMAP
                                   session; latency-to-first-detection

Outputs
-------
  results/ea_ed_results.json
  TIFS3_Submission/Figures/fig_ea_significance.pdf
  TIFS3_Submission/Figures/fig_eb_risk_coverage.pdf
  TIFS3_Submission/Figures/fig_ec_conformal_shift.pdf
  TIFS3_Submission/Figures/fig_ed_session_level.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, accuracy_score,
)
from sklearn.model_selection import train_test_split as _tts
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
CKPT    = ROOT / "checkpoints"
SPLITS  = ROOT / "splits"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "TIFS3_Submission" / "Figures"
RES_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANSMAP  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RANSKNOW = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"
RDSET    = "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet"

SPLIT_NAMES = ["random_stratified", "family_holdout", "hardware_holdout", "temporal_holdout"]
SPLIT_LABELS = ["Random", "Family", "Hardware", "Temporal"]
SEEDS   = [0, 1, 2, 3, 4]   # 42 uses the canonical checkpoints
T_MC    = 10
BATCH   = 512
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOWLEDGE_DROP = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_Count", "Family_List",
    "Tool_List", "Dominant_Tactic", "Platform_Signal",
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
DPI = 300

# ── model definitions (mirror eval_sabre2.py) ─────────────────────────────────
class CNNBehavioral(nn.Module):
    def __init__(self, in_dim=16, p=0.3):
        super().__init__()
        pooled = (in_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
            nn.Linear(pooled * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.model(x).squeeze(-1)

class CNNSemantic(nn.Module):
    def __init__(self, in_dim=384, p=0.3):
        super().__init__()
        pool_dim = (in_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
            nn.Linear(pool_dim * 32, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.model(x).squeeze(-1)

class CNNKnowledge(nn.Module):
    def __init__(self, in_dim=23, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.fc(self.conv(x)).squeeze(-1)

class FusionMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ── MC inference helpers ──────────────────────────────────────────────────────
def _enable_mc(m):
    if isinstance(m, nn.Dropout): m.train()

def mc_infer_batched(model, X: np.ndarray, T=T_MC):
    """MC Dropout inference — enables only Dropout modules, matching eval_sabre2.py."""
    model.to(DEVICE).eval()
    model.apply(_enable_mc)
    means, stds = [], []
    for i in range(0, len(X), BATCH):
        batch = torch.tensor(X[i:i+BATCH], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        passes = []
        with torch.no_grad():
            for _ in range(T):
                passes.append(model(batch).cpu().numpy())
        ps = np.stack(passes)
        means.append(ps.mean(0)); stds.append(ps.std(0))
    return np.concatenate(means), np.concatenate(stds)

def det_infer_batched(model, X: np.ndarray):
    """Deterministic eval-mode inference (no dropout) — matches multiseed_runner evaluate_model."""
    model.to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            batch = torch.tensor(X[i:i+BATCH], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            out.append(model(batch).cpu().numpy())
    return np.concatenate(out)

def fuse_mlp_forward(mlp, sc, T_cal, feat: np.ndarray) -> np.ndarray:
    Xsc = sc.transform(feat).astype(np.float32)
    with torch.no_grad():
        probs = mlp(torch.from_numpy(Xsc).to(DEVICE)).cpu().numpy()
    if T_cal != 1.0:
        eps = 1e-6
        p   = np.clip(probs, eps, 1 - eps)
        logits = np.log(p) - np.log(1 - p)
        probs  = 1 / (1 + np.exp(-logits / T_cal))
    return probs

EPS = 1e-6

def peer_consult(p_s, p_b, p_k, sig_s, sig_b, sig_k, kl=0.15):
    p_s, p_b, p_k = p_s.copy(), p_b.copy(), p_k.copy()
    for mask, own_s, peers in [
        (sig_s > kl, sig_s, [(p_b, sig_b), (p_k, sig_k)]),
        (sig_b > kl, sig_b, [(p_s, sig_s), (p_k, sig_k)]),
        (sig_k > kl, sig_k, [(p_s, sig_s), (p_b, sig_b)]),
    ]:
        if mask.any():
            ws  = [1 / (s[mask] + EPS) for _, s in peers]
            ws_sum = sum(ws)
            adj = sum(w * p[mask] for w, (p, _) in zip(ws, peers)) / ws_sum
            if own_s is sig_s: p_s[mask] = adj
            elif own_s is sig_b: p_b[mask] = adj
            else: p_k[mask] = adj
    return p_s, p_b, p_k

# ── loaders ───────────────────────────────────────────────────────────────────
META = json.loads((CKPT / "metadata.json").read_text())

def load_behavioral_canonical(split: str):
    dim = META["behavioral_input_dim"]
    m   = CNNBehavioral(dim)
    m.load_state_dict(torch.load(str(CKPT / f"behavioral_cnn_{split}.pt"), map_location="cpu"))
    m.eval()
    sc = joblib.load(str(CKPT / f"behavioral_scaler_{split}.joblib"))
    return m, sc

def load_behavioral_seed(seed: int, split: str):
    dim = META["behavioral_input_dim"]
    m   = CNNBehavioral(dim)
    m.load_state_dict(torch.load(str(CKPT / "seeds" / f"seed{seed}_{split}.pt"), map_location="cpu"))
    m.eval()
    sc = joblib.load(str(CKPT / "seeds" / f"seed{seed}_{split}_scaler.joblib"))
    return m, sc

def load_knowledge():
    dim = META["knowledge_input_dim"]
    m   = CNNKnowledge(dim)
    with torch.no_grad(): _ = m(torch.zeros(1, 1, dim))
    m.load_state_dict(torch.load(str(CKPT / "knowledge_cnn.pt"), map_location="cpu"))
    m.eval()
    sc = joblib.load(str(CKPT / "knowledge_scaler.joblib"))
    return m, sc

def load_semantic():
    dim = META["semantic_input_dim"]
    m   = CNNSemantic(dim)
    m.load_state_dict(torch.load(str(CKPT / "semantic_cnn.pt"), map_location="cpu"), strict=True)
    m.eval()
    return m

def load_decision_mlp():
    blob   = json.loads((CKPT / "decision_fuser_3agent.json").read_text())
    cfg    = blob["cfg"]
    mlp    = FusionMLP(blob["in_dim"], cfg["hidden"], cfg["p_drop"])
    state  = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["model_state"].items()}
    mlp.load_state_dict(state, strict=True)
    mlp.eval().to(DEVICE)
    sc     = StandardScaler()
    sc.mean_  = np.array(blob["scaler_mean"],  dtype=np.float32)
    sc.scale_ = np.array(blob["scaler_scale"], dtype=np.float32)
    T_cal  = float(np.exp(blob.get("temp_logT", 0.0)))
    return mlp, sc, T_cal

def load_ransmap(split: str, subset: str):
    sess = set((SPLITS / split / f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values
    sess_col = sub["session"].values
    return X, y, fam, sess_col

def build_oof_behavioural_scores(n_folds=5):
    """Out-of-fold Behavioural MC-Dropout scores for random_stratified's full
    train set (see eval_sabre2.py's build_oof_behavioural_scores for the full
    rationale — in-sample scoring is Decision-MLP stacking leakage).
    splits/cv_fold_{k} partition the train sessions into 5 disjoint
    session-level folds; checkpoints/behavioral_cnn_cv_fold_{k}.pt is trained
    on the other 4 folds only."""
    y_all, fam_all, p_all, sig_all = [], [], [], []
    for k in range(n_folds):
        beh_m_k, beh_sc_k = load_behavioral_canonical(f"cv_fold_{k}")
        X_k, y_k, fam_k, _ = load_ransmap(f"cv_fold_{k}", "test")
        Xs_k = beh_sc_k.transform(X_k).astype(np.float32)
        p_k, sig_k = mc_infer_batched(beh_m_k, Xs_k)
        y_all.append(y_k); fam_all.append(fam_k)
        p_all.append(p_k); sig_all.append(sig_k)
    return (np.concatenate(p_all), np.concatenate(sig_all),
            np.concatenate(y_all).astype(int), np.concatenate(fam_all))

def build_knowledge_lookup(know_m, know_sc):
    """Return a label-independent corpus-mean Knowledge vector for every RanSMAP
    sample, regardless of its family (see eval_sabre2.py's build_knowledge_lookup
    for the full rationale: no real per-sample RansKnow/RanSMAP pairing exists,
    so a per-family lookup keyed by the sample's own true label would leak it)."""
    df   = pd.read_csv(RANSKNOW)
    fc   = META["knowledge_features"]
    Xrk  = know_sc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    corp = Xrk.mean(0)
    def lookup(cn: str):
        return corp
    return lookup

def build_semantic_fill(sem_m, mlp_sc):
    df  = pd.read_parquet(RDSET)
    emb = np.stack(df["embedding"].values).mean(axis=0, keepdims=True).astype(np.float32)
    p_s, sig_s = mc_infer_batched(sem_m, emb, T=T_MC)
    return float(p_s[0]), float(mlp_sc.mean_[3])  # σ_s from scaler mean

def train_fresh_mlp(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0):
    """Train FusionMLP on 6D score features (no CTGAN augmentation)."""
    sc = StandardScaler().fit(X_tr)
    X_sc = sc.transform(X_tr).astype(np.float32)
    y_f  = y_tr.astype(np.float32)
    X_t, X_v, y_t, y_v = _tts(
        X_sc, y_f, test_size=0.2,
        stratify=(y_f > 0.5).astype(int), random_state=seed)
    torch.manual_seed(seed)
    net = FusionMLP(6, 32, 0.2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_t), torch.tensor(y_t)),
        batch_size=128, shuffle=True)
    best_va, best_state, pat = float("inf"), None, 0
    for _ in range(40):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); bce(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = bce(net(torch.tensor(X_v).to(DEVICE)),
                     torch.tensor(y_v).to(DEVICE)).item()
        if va + 1e-6 < best_va:
            best_va = va; pat = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 6: break
    net.load_state_dict(best_state); net.eval()
    return net, sc

def score_split(split, subset, beh_m, beh_sc, know_m, know_sc,
                know_lookup, mlp, mlp_sc, T_cal, sem_fill, use_pc=True):
    p_s_fill, sig_s_fill = sem_fill
    X, y, fam, sessions  = load_ransmap(split, subset)
    Xs   = beh_sc.transform(X).astype(np.float32)
    p_b, sig_b = mc_infer_batched(beh_m, Xs)
    Xk   = np.stack([know_lookup(f) for f in fam]).astype(np.float32)
    p_k, sig_k = mc_infer_batched(know_m, Xk)
    n    = len(p_b)
    p_s  = np.full(n, p_s_fill, dtype=np.float32)
    sig_s = np.full(n, sig_s_fill, dtype=np.float32)
    p_b_raw = p_b.copy()   # pre-PC behavioral score (standalone B metric)
    if use_pc:
        p_s, p_b, p_k = peer_consult(p_s, p_b, p_k, sig_s, sig_b, sig_k)
    feat       = np.stack([p_s, p_b, p_k, sig_s, sig_b, sig_k], axis=1).astype(np.float32)
    probs_mlp  = fuse_mlp_forward(mlp, mlp_sc, T_cal, feat)
    return dict(y=y, fam=fam, sessions=sessions,
                p_b=p_b_raw, sig_b=sig_b, probs_mlp=probs_mlp,
                sig_max=np.stack([sig_b, sig_k]).max(0))


# ═════════════════════════════════════════════════════════════════════════════
# Shared model loading (done once)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading fixed models …")
know_m, know_sc = load_knowledge()
sem_m           = load_semantic()
know_lookup     = build_knowledge_lookup(know_m, know_sc)

# σ_s fill uses the stored scaler's mean_[3] (same as eval_sabre2.py line 419)
_blob   = json.loads((CKPT / "decision_fuser_3agent.json").read_text())
_sc_ref = StandardScaler()
_sc_ref.mean_  = np.array(_blob["scaler_mean"],  dtype=np.float32)
_sc_ref.scale_ = np.array(_blob["scaler_scale"], dtype=np.float32)
print("  Building semantic fill …")
sem_fill = build_semantic_fill(sem_m, _sc_ref)
print(f"  Semantic fill: p_s={sem_fill[0]:.4f}  σ_s={sem_fill[1]:.2e}")

# Build 6D training score features, out-of-fold (no stacking leak) —
# mirrors eval_sabre2.py's build_oof_behavioural_scores
print("\nBuilding 6D training score features (random_stratified train split, out-of-fold) …")
p_b_tr, sig_b_tr, y_tr_raw, fam_tr = build_oof_behavioural_scores()
Xk_tr   = np.stack([know_lookup(f) for f in fam_tr]).astype(np.float32)
p_k_tr, sig_k_tr = mc_infer_batched(know_m, Xk_tr)
n_tr    = len(p_b_tr)
p_s_tr  = np.full(n_tr, sem_fill[0], np.float32)
sig_s_tr = np.full(n_tr, sem_fill[1], np.float32)
p_s_tr_pc, p_b_tr_pc, p_k_tr_pc = peer_consult(
    p_s_tr, p_b_tr, p_k_tr, sig_s_tr, sig_b_tr, sig_k_tr)
X_tr_score = np.stack([p_s_tr_pc, p_b_tr_pc, p_k_tr_pc,
                       sig_s_tr, sig_b_tr, sig_k_tr], axis=1).astype(np.float32)

# Train a fresh Decision MLP (no CTGAN, seed=0) — matches eval_sabre2.py training recipe
print("Training fresh Decision MLP (no augmentation, seed=0) …")
mlp, mlp_sc = train_fresh_mlp(X_tr_score, y_tr_raw.astype(np.float32), seed=0)
T_cal = 1.0

# Canonical (seed-42 behavioral CNN) inference on all 4 splits — used by E-B, E-C, E-D
print("\nRunning canonical (seed-42) inference on all splits …")
canonical = {}
for sp in SPLIT_NAMES:
    print(f"  {sp} …")
    beh_m, beh_sc = load_behavioral_canonical(sp)
    canonical[sp] = score_split(sp, "test", beh_m, beh_sc, know_m, know_sc,
                                know_lookup, mlp, mlp_sc, T_cal, sem_fill)
    y_sp = canonical[sp]["y"]
    print(f"    B AUROC={roc_auc_score(y_sp, canonical[sp]['p_b']):.4f}  "
          f"MLP AUROC={roc_auc_score(y_sp, canonical[sp]['probs_mlp']):.4f}")

results: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# E-A  Statistical significance
#      One-sample Wilcoxon: per-seed B AUROC (n=6) vs reported MLP AUROC ref
#      McNemar: per-sample predictions (canonical B vs fresh MLP, random_stratified)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-A  Statistical Significance")
print("="*65)

# Per-seed B AUROCs from multiseed.json (deterministic eval mode, authoritative)
_ms    = json.loads((RES_DIR / "multiseed.json").read_text())
_sabre = json.loads((RES_DIR / "sabre2_eval.json").read_text())
seed_b_by_split: dict = {sp: [] for sp in SPLIT_NAMES}
for row in _ms["per_run"]:
    seed_b_by_split[row["split"]].append(row["test_auroc"])

ea_results = {}
# PRIMARY test: B is a single fixed checkpoint at deployment (it is not
# retrained per seed), so the only genuine source of paired randomness is
# the Decision MLP's own training (data shuffling, weight init, CTGAN
# sampling) across K=6 seeds. We therefore run a one-sample Wilcoxon of the
# 6 per-seed MLP AUROCs against the single, fixed, deployed B AUROC --
# this is the correct one-sample design (previously the roles were
# reversed: a single MLP reference value was compared against B's
# unrelated multiseed *retraining* noise, which is not what either agent
# actually varies over at deployment).
print("\n  One-sample Wilcoxon (MLP per-seed vs fixed deployed B, H1: MLP > B):")
for sp in SPLIT_NAMES:
    mlp_seeds = np.array(_sabre["E1_main"][sp]["BK_mlp"]["auroc_seeds"])
    b_fixed   = _sabre["E1_main"][sp]["B"]["auroc"]
    diff      = mlp_seeds - b_fixed  # positive where MLP > fixed B
    try:
        stat, pval = wilcoxon(diff, alternative="greater")
    except Exception:
        stat, pval = float("nan"), 1.0  # all zeros → indeterminate

    # SECONDARY robustness check: does the same conclusion hold if we
    # instead account for B's own retraining variance (6 independently
    # retrained Behavioural checkpoints from multiseed.json), paired
    # index-for-index against the 6 MLP seeds? Retraining seed and MLP
    # seed are not causally linked, so this is a robustness check, not the
    # primary test.
    b_multiseed = np.array(seed_b_by_split[sp])
    n_pair = min(len(mlp_seeds), len(b_multiseed))
    diff_paired = mlp_seeds[:n_pair] - b_multiseed[:n_pair]
    try:
        stat_p, pval_p = wilcoxon(diff_paired, alternative="greater")
    except Exception:
        stat_p, pval_p = float("nan"), 1.0

    ea_results[sp] = {
        "B_fixed_auroc": b_fixed,
        "MLP_seeds": mlp_seeds.round(4).tolist(),
        "MLP_mean":  float(mlp_seeds.mean()),
        "MLP_ci95_lo": _sabre["E1_main"][sp]["BK_mlp"]["auroc"]["ci95_lo"],
        "MLP_ci95_hi": _sabre["E1_main"][sp]["BK_mlp"]["auroc"]["ci95_hi"],
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "wilcoxon_p":    float(pval),
        "note": "PRIMARY: one-sample, diff = MLP_seed_i - fixed_deployed_B; H1: median diff > 0",
        "robustness_check_vs_B_retraining_noise": {
            "B_multiseed_mean": float(b_multiseed[:n_pair].mean()),
            "B_multiseed_std":  float(b_multiseed[:n_pair].std(ddof=1)),
            "B_multiseed_seeds": [round(v, 4) for v in b_multiseed[:n_pair].tolist()],
            "wilcoxon_stat": float(stat_p) if not np.isnan(stat_p) else None,
            "wilcoxon_p":    float(pval_p),
            "note": "SECONDARY: paired index-for-index against B's own retraining noise (not causally linked to MLP seed)",
        },
    }
    print(f"  {sp:25s}  B_fixed={b_fixed:.4f}  MLP={mlp_seeds.mean():.4f}±{mlp_seeds.std(ddof=1):.4f}  "
          f"Δ={diff.mean():.4f}  p={pval:.4f}  (robustness p={pval_p:.4f})")

# McNemar on random_stratified, WINDOW level. Windows from the same session
# are correlated (they share one execution's telemetry characteristics), so
# this violates McNemar's independence assumption and is reported only for
# continuity with earlier drafts -- the session-level test below is primary.
d_rs      = canonical["random_stratified"]
preds_b   = (d_rs["p_b"]       >= 0.5).astype(int)
preds_mlp = (d_rs["probs_mlp"] >= 0.5).astype(int)
y_rs      = d_rs["y"]
b_right_mlp_wrong = int(((preds_b == y_rs) & (preds_mlp != y_rs)).sum())
b_wrong_mlp_right = int(((preds_b != y_rs) & (preds_mlp == y_rs)).sum())
b_mc, c_mc = b_right_mlp_wrong, b_wrong_mlp_right
mc_chi2 = (abs(b_mc - c_mc) - 1) ** 2 / (b_mc + c_mc + 1e-9)
mc_pval = float(chi2.sf(mc_chi2, df=1))
ea_results["mcnemar_window_level_random_stratified"] = {
    "note": "NOT independent -- windows within a session are correlated; see mcnemar_session_level below",
    "B_right_MLP_wrong": b_mc,
    "B_wrong_MLP_right": c_mc,
    "chi2":   round(float(mc_chi2), 4),
    "pvalue": round(mc_pval, 6),
}
print(f"\n  McNemar, window-level, NOT independent (random_stratified): "
      f"B_right/MLP_wrong={b_mc}  B_wrong/MLP_right={c_mc}  p={mc_pval:.4f}")

# McNemar on random_stratified, SESSION level (majority-vote aggregation,
# same rule as E-D) -- this is the independence-respecting test.
sessions_rs = d_rs["sessions"]
uniq_sessions = np.unique(sessions_rs)
sess_y, sess_pred_b, sess_pred_mlp = [], [], []
for s in uniq_sessions:
    m = sessions_rs == s
    sess_y.append(int(y_rs[m][0]))
    sess_pred_b.append(int(preds_b[m].mean() > 0.5))
    sess_pred_mlp.append(int(preds_mlp[m].mean() > 0.5))
sess_y        = np.array(sess_y)
sess_pred_b   = np.array(sess_pred_b)
sess_pred_mlp = np.array(sess_pred_mlp)
b_right_mlp_wrong_sess = int(((sess_pred_b == sess_y) & (sess_pred_mlp != sess_y)).sum())
b_wrong_mlp_right_sess = int(((sess_pred_b != sess_y) & (sess_pred_mlp == sess_y)).sum())
b_s, c_s = b_right_mlp_wrong_sess, b_wrong_mlp_right_sess
mc_chi2_sess = (abs(b_s - c_s) - 1) ** 2 / (b_s + c_s + 1e-9)
mc_pval_sess = float(chi2.sf(mc_chi2_sess, df=1))
ea_results["mcnemar_session_level_random_stratified"] = {
    "n_sessions": int(len(uniq_sessions)),
    "B_right_MLP_wrong": b_s,
    "B_wrong_MLP_right": c_s,
    "chi2":   round(float(mc_chi2_sess), 4),
    "pvalue": round(mc_pval_sess, 6),
}
print(f"  McNemar, session-level (n={len(uniq_sessions)}, random_stratified): "
      f"B_right/MLP_wrong={b_s}  B_wrong/MLP_right={c_s}  p={mc_pval_sess:.4f}")

results["E_A"] = ea_results

# Figure: per-split Decision MLP (boxplot, 6 seeds -- the actual paired
# randomness) vs the single fixed, deployed Behavioural CNN (point, no CI
# since it is deterministic at deployment).
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(6.0, 3.2))
x = np.arange(len(SPLIT_NAMES))
w = 0.30
c_b = "#2c7bb6"; c_m = "#d7191c"
bp_m = ax.boxplot([ea_results[sp]["MLP_seeds"] for sp in SPLIT_NAMES],
                   positions=x + w/2, widths=w*0.9,
                   patch_artist=True, boxprops=dict(facecolor=c_m, alpha=0.6),
                   medianprops=dict(color="darkred", lw=1.5),
                   whiskerprops=dict(lw=1), capprops=dict(lw=1),
                   flierprops=dict(marker="x", markersize=4))
for i, sp in enumerate(SPLIT_NAMES):
    r = ea_results[sp]
    ax.plot(x[i] - w/2, r["B_fixed_auroc"], marker="D", color=c_b, markersize=6)
    pval = r["wilcoxon_p"]
    sig  = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    ymax = max(max(r["MLP_seeds"]), r["B_fixed_auroc"]) + 0.012
    ax.text(i, ymax, sig, ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(SPLIT_LABELS)
ax.set_ylabel("AUROC")
ax.set_title("Decision MLP (boxplot, 6 training seeds) vs fixed, deployed Behavioural CNN (point)\n"
             "One-sample Wilcoxon (H1: MLP > B): * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
             fontsize=7.5)
ax.legend(handles=[Patch(facecolor=c_m, alpha=0.6, label="Decision MLP (6 training seeds)"),
                   Patch(facecolor=c_b, alpha=0.8, label="Behavioural CNN (fixed, deployed)")],
          fontsize=7, loc="lower left")
ax.set_ylim(0.55, 1.04)
plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_ea_significance.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_ea_significance.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\n  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# E-B  Risk-coverage curve (AURC)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-B  Risk-Coverage (AURC)")
print("="*65)

def risk_coverage_curve(probs, y, n_thresh=200):
    """
    Auto-Contain coverage: fraction of samples classified as ransomware
    with confidence >= tau.  Risk = 1 - precision on that subset.
    Returns (coverage, precision) arrays and AURC.
    """
    thresholds = np.linspace(1.0, 0.0, n_thresh + 1)
    coverages  = []
    precisions = []
    for tau in thresholds:
        mask = probs >= tau
        cov  = mask.mean()
        if mask.sum() == 0:
            prec = 1.0
        else:
            prec = (y[mask] == 1).mean()
        coverages.append(float(cov))
        precisions.append(float(prec))
    coverages  = np.array(coverages)
    precisions = np.array(precisions)
    # AURC = area under precision-coverage curve (trapz, coverage ascending)
    order = np.argsort(coverages)
    aurc  = float(np.trapz(precisions[order], coverages[order]))
    return coverages, precisions, aurc

eb_results = {}
fig, axes  = plt.subplots(1, 2, figsize=(7.0, 3.0))
colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]

ax1, ax2 = axes
for i, (sp, lbl) in enumerate(zip(SPLIT_NAMES, SPLIT_LABELS)):
    d  = canonical[sp]
    cov, prec, aurc = risk_coverage_curve(d["probs_mlp"], d["y"])
    eb_results[sp] = {"aurc": round(aurc, 4)}
    ax1.plot(cov, prec, color=colors[i], linewidth=1.3,
             label=f"{lbl} (AURC={aurc:.3f})")
    print(f"  {sp:25s}  AURC={aurc:.4f}")

ax1.set_xlabel("Auto-Contain Coverage")
ax1.set_ylabel("Precision on Auto-Contained")
ax1.set_title("Risk-Coverage Curve (Decision MLP)", fontsize=9, fontweight="bold")
ax1.legend(fontsize=6.5, loc="lower left")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
ax1.axhline(1.0, color="gray", ls="--", lw=0.8, label="Perfect")

# Also plot AUROC vs AURC side by side as bar chart
split_short = SPLIT_LABELS
auroc_vals  = [roc_auc_score(canonical[sp]["y"], canonical[sp]["probs_mlp"]) for sp in SPLIT_NAMES]
aurc_vals   = [eb_results[sp]["aurc"] for sp in SPLIT_NAMES]
x_pos       = np.arange(len(SPLIT_NAMES))
ax2.bar(x_pos - 0.2, auroc_vals, 0.35, label="AUROC", color="#2c7bb6", alpha=0.8)
ax2.bar(x_pos + 0.2, aurc_vals,  0.35, label="AURC",  color="#d7191c", alpha=0.8)
for xi, (a, r) in enumerate(zip(auroc_vals, aurc_vals)):
    ax2.text(xi - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=6.5)
    ax2.text(xi + 0.2, r + 0.005, f"{r:.3f}", ha="center", fontsize=6.5)
ax2.set_xticks(x_pos); ax2.set_xticklabels(split_short)
ax2.set_ylabel("Metric Value")
ax2.set_title("AUROC vs AURC per Split", fontsize=9, fontweight="bold")
ax2.legend(fontsize=7)
ax2.set_ylim(0, 1.12)

plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_eb_risk_coverage.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_eb_risk_coverage.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved {out}")

results["E_B"] = eb_results


# ═════════════════════════════════════════════════════════════════════════════
# E-C  Conformal coverage under distribution shift
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-C  Conformal Coverage under Shift")
print("="*65)

ALPHAS = [0.01, 0.05, 0.10]  # miscoverage rates → 99%, 95%, 90% nominal

def mondrian_thresholds(probs_cal, y_cal, alpha):
    idx1    = y_cal == 1; idx0 = y_cal == 0
    scores1 = 1.0 - probs_cal[idx1]
    scores0 = probs_cal[idx0]
    n1, n0  = idx1.sum(), idx0.sum()
    q1 = float(np.quantile(scores1, min(1.0, np.ceil((n1 + 1) * (1 - alpha)) / n1)))
    q0 = float(np.quantile(scores0, min(1.0, np.ceil((n0 + 1) * (1 - alpha)) / n0)))
    return q1, q0

def conformal_metrics(probs_test, y_test, q1, q0):
    inc1     = probs_test >= 1 - q1
    inc0     = probs_test <= q0
    detect   = inc1 & ~inc0
    benign   = inc0 & ~inc1
    escalate = inc1 & inc0
    auto_cls = detect | benign
    correct  = (y_test * inc1 + (1 - y_test) * inc0).astype(bool)
    tp = int((detect & (y_test == 1)).sum())
    fp = int((detect & (y_test == 0)).sum())
    return {
        "empirical_coverage": float(correct.mean()),
        "detect_rate":        float(detect.mean()),
        "benign_rate":        float(benign.mean()),
        "escalate_rate":      float(escalate.mean()),
        "auto_classify_rate": float(auto_cls.mean()),
        "detect_precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }

ec_results = {}
for sp in SPLIT_NAMES:
    print(f"  {sp} …")
    beh_m_sp, beh_sc_sp = load_behavioral_canonical(sp)
    val_data  = score_split(sp, "val",  beh_m_sp, beh_sc_sp, know_m, know_sc,
                            know_lookup, mlp, mlp_sc, T_cal, sem_fill)
    test_data = canonical[sp]   # already scored with same canonical model
    ec_results[sp] = {}
    for alpha in ALPHAS:
        q1, q0 = mondrian_thresholds(val_data["probs_mlp"], val_data["y"], alpha)
        m      = conformal_metrics(test_data["probs_mlp"], test_data["y"], q1, q0)
        m["nominal_coverage"] = round(1 - alpha, 2)
        m["alpha"]            = alpha
        ec_results[sp][f"alpha_{alpha}"] = m
        print(f"    α={alpha}  nominal={1-alpha:.2f}  empirical={m['empirical_coverage']:.4f}"
              f"  escalate={m['escalate_rate']:.3f}  precision={m['detect_precision']:.4f}")

results["E_C"] = ec_results

# Figure: heatmap of empirical coverage vs nominal across splits
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
alpha_labels   = [f"α={a}\n({int((1-a)*100)}% nominal)" for a in ALPHAS]
cov_matrix     = np.array([[ec_results[sp][f"alpha_{a}"]["empirical_coverage"]
                             for a in ALPHAS] for sp in SPLIT_NAMES])
esc_matrix     = np.array([[ec_results[sp][f"alpha_{a}"]["escalate_rate"]
                             for a in ALPHAS] for sp in SPLIT_NAMES])

im1 = axes[0].imshow(cov_matrix, vmin=0.85, vmax=1.0, cmap="RdYlGn", aspect="auto")
for i in range(len(SPLIT_NAMES)):
    for j in range(len(ALPHAS)):
        v   = cov_matrix[i, j]
        nom = 1 - ALPHAS[j]
        axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                     color="black" if abs(v - nom) < 0.02 else "navy")
axes[0].set_xticks(range(len(ALPHAS))); axes[0].set_xticklabels(alpha_labels)
axes[0].set_yticks(range(len(SPLIT_NAMES))); axes[0].set_yticklabels(SPLIT_LABELS)
axes[0].set_title("Empirical Coverage\n(dashed = nominal)", fontsize=8.5, fontweight="bold")
plt.colorbar(im1, ax=axes[0], fraction=0.04, pad=0.03)
# Overlay nominal coverage as horizontal lines on colorbar? Just annotate
for j, a in enumerate(ALPHAS):
    axes[0].axvline(j - 0.5, color="gray", lw=0.3, ls=":")

im2 = axes[1].imshow(esc_matrix, vmin=0.0, vmax=0.5, cmap="YlOrRd", aspect="auto")
for i in range(len(SPLIT_NAMES)):
    for j in range(len(ALPHAS)):
        axes[1].text(j, i, f"{esc_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)
axes[1].set_xticks(range(len(ALPHAS))); axes[1].set_xticklabels(alpha_labels)
axes[1].set_yticks(range(len(SPLIT_NAMES))); axes[1].set_yticklabels(SPLIT_LABELS)
axes[1].set_title("Escalation Rate", fontsize=8.5, fontweight="bold")
plt.colorbar(im2, ax=axes[1], fraction=0.04, pad=0.03)

plt.suptitle("Mondrian Conformal Prediction — Empirical vs Nominal Coverage\n"
             "(calibrated per-split on val set; tested on held-out test set)",
             fontsize=8, y=1.01)
plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_ec_conformal_shift.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_ec_conformal_shift.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# E-D  Session-level aggregation
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-D  Session-level Aggregation")
print("="*65)

WINDOW_SECS    = 10
THRESHOLD_SESS = 0.5

def session_metrics(y, probs, sessions):
    df = pd.DataFrame({"y": y, "prob": probs, "session": sessions})
    # Preserve window order within each session via iloc order (parquet row order)
    df["window_idx"] = df.groupby("session").cumcount()
    df["pred"] = (df["prob"] >= THRESHOLD_SESS).astype(int)

    rows = []
    for sess, grp in df.groupby("session", sort=False):
        grp     = grp.sort_values("window_idx").reset_index(drop=True)
        true_y  = int(grp["y"].iloc[0])          # all windows same label
        # Max-score rule: session positive if max(prob) >= 0.5
        max_prob = grp["prob"].max()
        max_pred = int(max_prob >= THRESHOLD_SESS)
        # Majority-vote rule
        maj_pred = int(grp["pred"].mean() >= 0.5)
        # Latency-to-first-detection (windows until first prob >= 0.5)
        first_det = grp.index[grp["pred"] == 1].min() if grp["pred"].any() else None
        latency_s = None if first_det is None else int(first_det) * WINDOW_SECS
        rows.append({"session": sess, "true_y": true_y,
                     "max_pred": max_pred, "maj_pred": maj_pred,
                     "max_prob": float(max_prob), "n_windows": len(grp),
                     "latency_s": latency_s})

    sess_df = pd.DataFrame(rows)
    results_out = {}
    for rule, col in [("max_score", "max_pred"), ("majority_vote", "maj_pred")]:
        preds = sess_df[col].values
        ytrue = sess_df["true_y"].values
        tp = int(((preds == 1) & (ytrue == 1)).sum())
        fp = int(((preds == 1) & (ytrue == 0)).sum())
        tn = int(((preds == 0) & (ytrue == 0)).sum())
        fn = int(((preds == 0) & (ytrue == 1)).sum())
        det_rate  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        balacc    = balanced_accuracy_score(ytrue, preds)
        results_out[rule] = {
            "detection_rate": round(float(det_rate), 4),
            "false_positive_rate": round(float(fpr), 4),
            "balanced_accuracy":   round(float(balacc), 4),
            "n_sessions": int(len(sess_df)),
            "n_ransomware_sessions": int((ytrue == 1).sum()),
            "n_benign_sessions":     int((ytrue == 0).sum()),
        }
    # Latency stats on ransomware sessions that were detected
    ransom_detected = sess_df[(sess_df["true_y"] == 1) & (sess_df["latency_s"].notna())]
    latencies = ransom_detected["latency_s"].values.astype(float)
    results_out["latency_to_first_detection_s"] = {
        "n_detected":   int(len(latencies)),
        "median":       float(np.median(latencies)) if len(latencies) else None,
        "mean":         float(np.mean(latencies))   if len(latencies) else None,
        "p10":          float(np.percentile(latencies, 10)) if len(latencies) else None,
        "p90":          float(np.percentile(latencies, 90)) if len(latencies) else None,
    }
    return results_out, sess_df

ed_results = {}
all_sess_dfs = {}
for sp in SPLIT_NAMES:
    d = canonical[sp]
    print(f"  {sp} …")
    res_sp, sdf = session_metrics(d["y"], d["probs_mlp"], d["sessions"])
    ed_results[sp] = res_sp
    all_sess_dfs[sp] = sdf
    lat = res_sp["latency_to_first_detection_s"]
    for rule in ["max_score", "majority_vote"]:
        r = res_sp[rule]
        print(f"    [{rule}]  det={r['detection_rate']:.3f}  FPR={r['false_positive_rate']:.3f}"
              f"  bal_acc={r['balanced_accuracy']:.3f}  n_sess={r['n_sessions']}")
    print(f"    [latency]  median={lat['median']}s  p10={lat['p10']}s  p90={lat['p90']}s")

results["E_D"] = ed_results

# Figure: session-level detection rate and FPR per split + latency distribution
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
c_max = "#2c7bb6"; c_maj = "#d7191c"
x    = np.arange(len(SPLIT_NAMES))
w    = 0.35

det_max = [ed_results[sp]["max_score"]["detection_rate"]   for sp in SPLIT_NAMES]
det_maj = [ed_results[sp]["majority_vote"]["detection_rate"] for sp in SPLIT_NAMES]
fpr_max = [ed_results[sp]["max_score"]["false_positive_rate"]   for sp in SPLIT_NAMES]
fpr_maj = [ed_results[sp]["majority_vote"]["false_positive_rate"] for sp in SPLIT_NAMES]

axes[0].bar(x - w/2, det_max, w, label="Max-score",    color=c_max, alpha=0.8)
axes[0].bar(x + w/2, det_maj, w, label="Majority-vote",color=c_maj, alpha=0.8)
for xi, (a, b) in enumerate(zip(det_max, det_maj)):
    axes[0].text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=6.5)
    axes[0].text(xi + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=6.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(SPLIT_LABELS)
axes[0].set_ylabel("Session Detection Rate")
axes[0].set_ylim(0, 1.15)
axes[0].set_title("Session-Level Detection Rate", fontsize=8.5, fontweight="bold")
axes[0].legend(fontsize=7)

axes[1].bar(x - w/2, fpr_max, w, label="Max-score",    color=c_max, alpha=0.8)
axes[1].bar(x + w/2, fpr_maj, w, label="Majority-vote",color=c_maj, alpha=0.8)
for xi, (a, b) in enumerate(zip(fpr_max, fpr_maj)):
    axes[1].text(xi - w/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=6.5)
    axes[1].text(xi + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=6.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(SPLIT_LABELS)
axes[1].set_ylabel("Session FPR")
axes[1].set_ylim(0, 0.5)
axes[1].set_title("Session-Level FPR", fontsize=8.5, fontweight="bold")
axes[1].legend(fontsize=7)

# Latency histogram for random_stratified
lat_vals = []
sdf_rs = all_sess_dfs["random_stratified"]
lat_rs = sdf_rs[(sdf_rs["true_y"] == 1) & sdf_rs["latency_s"].notna()]["latency_s"].values
axes[2].hist(lat_rs, bins=20, color=c_max, alpha=0.75, edgecolor="white")
if len(lat_rs):
    axes[2].axvline(np.median(lat_rs), color="navy", ls="--", lw=1.5,
                    label=f"Median {int(np.median(lat_rs))}s")
axes[2].set_xlabel("Latency to First Detection (s)")
axes[2].set_ylabel("Count")
axes[2].set_title("Time to First Detection\n(Random-Stratified, ransomware sessions)",
                   fontsize=8.5, fontweight="bold")
axes[2].legend(fontsize=7)

plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_ed_session_level.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_ed_session_level.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Save results JSON
# ═════════════════════════════════════════════════════════════════════════════
out_json = RES_DIR / "ea_ed_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {out_json}")
print("Done.")
