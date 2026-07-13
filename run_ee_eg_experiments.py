#!/usr/bin/env python3
"""
run_ee_eg_experiments.py
========================
E-E  Stronger fusion baselines   RF, GBT (XGBoost substitute), MLP-2L
E-F  CTGAN ablation under severe imbalance   1%, 5%, 10% ransomware prevalence
E-G  Repeated leave-one-family-out   6 ransomware families

Outputs
-------
  results/ee_eg_results.json
  TIFS3_Submission/Figures/fig_ee_baselines.pdf
  TIFS3_Submission/Figures/fig_ef_ctgan_ablation.pdf
  TIFS3_Submission/Figures/fig_eg_lofo.pdf
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split as _tts
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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

SPLIT_NAMES  = ["random_stratified", "family_holdout", "hardware_holdout", "temporal_holdout"]
SPLIT_LABELS = ["Random", "Family", "Hardware", "Temporal"]
RANSOM_FAMILIES = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]
T_MC   = 10
BATCH  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
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

class FusionMLP2L(nn.Module):
    def __init__(self, in_dim=6, h1=32, h2=16, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ── MC inference helpers ──────────────────────────────────────────────────────
def _enable_mc(m):
    if isinstance(m, nn.Dropout): m.train()

def mc_infer_batched(model, X: np.ndarray, T=T_MC):
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

def load_ransmap(split: str, subset: str):
    sess = set((SPLITS / split / f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values
    return X, y, fam

def build_knowledge_lookup(know_m, know_sc):
    """Return a label-independent corpus-mean Knowledge vector for every RanSMAP
    sample, regardless of its family (see eval_sabre2.py's build_knowledge_lookup
    for the full rationale: no real per-sample RansKnow/RanSMAP pairing exists,
    so a per-family lookup keyed by the sample's own true label would leak it —
    this also removes the LOFO E-G leak, since the "held-out" family no longer
    gets a family-specific Knowledge score at all)."""
    df  = pd.read_csv(RANSKNOW)
    fc  = META["knowledge_features"]
    Xrk = know_sc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    corp = Xrk.mean(0)
    def lookup(cn: str):
        return corp
    return lookup

def build_semantic_fill(sem_m, mlp_sc_ref):
    df  = pd.read_parquet(RDSET)
    emb = np.stack(df["embedding"].values).mean(axis=0, keepdims=True).astype(np.float32)
    p_s, _ = mc_infer_batched(sem_m, emb, T=T_MC)
    return float(p_s[0]), float(mlp_sc_ref.mean_[3])

def score_features(X_beh, fam, beh_m, beh_sc, know_m, know_lookup, sem_fill):
    """Build 6D score feature matrix with peer-consult."""
    p_s_fill, sig_s_fill = sem_fill
    Xs   = beh_sc.transform(X_beh).astype(np.float32)
    p_b, sig_b = mc_infer_batched(beh_m, Xs)
    Xk   = np.stack([know_lookup(f) for f in fam]).astype(np.float32)
    p_k, sig_k = mc_infer_batched(know_m, Xk)
    n    = len(p_b)
    p_s  = np.full(n, p_s_fill, np.float32)
    sig_s = np.full(n, sig_s_fill, np.float32)
    p_s_pc, p_b_pc, p_k_pc = peer_consult(p_s, p_b, p_k, sig_s, sig_b, sig_k)
    feat = np.stack([p_s_pc, p_b_pc, p_k_pc, sig_s, sig_b, sig_k], axis=1).astype(np.float32)
    return feat

def build_oof_train_score_features(know_m, know_lookup, sem_fill, n_folds=5):
    """Out-of-fold 6D training score features for random_stratified's full
    train set (see eval_sabre2.py's build_oof_behavioural_scores for the full
    rationale: scoring a Behavioural checkpoint on its own training data is
    Decision-MLP stacking leakage). splits/cv_fold_{k} partition the train
    sessions into 5 disjoint session-level folds; checkpoints for fold k are
    trained on the other 4 folds only. Knowledge/Semantic are already
    family-agnostic constants, so reusing them across folds adds no leakage."""
    feats, ys, fams = [], [], []
    for k in range(n_folds):
        beh_m_k, beh_sc_k = load_behavioral_canonical(f"cv_fold_{k}")
        X_k, y_k, fam_k = load_ransmap(f"cv_fold_{k}", "test")
        feat_k = score_features(X_k, fam_k, beh_m_k, beh_sc_k, know_m, know_lookup, sem_fill)
        feats.append(feat_k); ys.append(y_k); fams.append(fam_k)
    return np.concatenate(feats), np.concatenate(ys).astype(int), np.concatenate(fams)

# ── MLP training ──────────────────────────────────────────────────────────────
def _train_net(net, X_tr, y_tr, seed=0, epochs=40, patience=6):
    sc = StandardScaler().fit(X_tr)
    X_sc = sc.transform(X_tr).astype(np.float32)
    y_f  = y_tr.astype(np.float32)
    X_t, X_v, y_t, y_v = _tts(
        X_sc, y_f, test_size=0.2,
        stratify=(y_f > 0.5).astype(int), random_state=seed)
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
            va = bce(net(torch.tensor(X_v).to(DEVICE)),
                     torch.tensor(y_v).to(DEVICE)).item()
        if va + 1e-6 < best_va:
            best_va = va; pat = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= patience: break
    net.load_state_dict(best_state); net.eval()
    return net, sc

def train_mlp1l(X_tr, y_tr, seed=0):
    net = FusionMLP(6, 32, 0.2).to(DEVICE)
    torch.manual_seed(seed)
    return _train_net(net, X_tr, y_tr, seed)

def train_mlp2l(X_tr, y_tr, seed=0):
    net = FusionMLP2L(6, 32, 16, 0.2).to(DEVICE)
    torch.manual_seed(seed)
    return _train_net(net, X_tr, y_tr, seed)

def predict_mlp(net, sc, X):
    Xs = sc.transform(X).astype(np.float32)
    with torch.no_grad():
        return net(torch.tensor(Xs).to(DEVICE)).cpu().numpy()

# ── ECE ───────────────────────────────────────────────────────────────────────
def ece_score(y_true, y_prob, n_bins=15):
    try:
        fracs, means = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        weights = counts / (counts.sum() + 1e-9)
        return float(np.sum(weights[:len(fracs)] * np.abs(fracs - means)))
    except Exception:
        return float("nan")

# ── SMOTE (NearestNeighbors interpolation) ────────────────────────────────────
def smote_oversample(X, y, seed=0):
    rng = np.random.default_rng(seed)
    min_idx = np.where(y == 1)[0]
    n_needed = int((y == 0).sum()) - int((y == 1).sum())
    if n_needed <= 0 or len(min_idx) < 2:
        return X, y
    k = min(5, len(min_idx) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X[min_idx])
    _, idxs = nbrs.kneighbors(X[min_idx])
    rows = []
    for _ in range(n_needed):
        i  = int(rng.integers(len(min_idx)))
        nb = int(idxs[i, rng.integers(1, idxs.shape[1])])
        lam = float(rng.random())
        rows.append(X[min_idx[i]] * lam + X[min_idx[nb]] * (1 - lam))
    X_syn = np.stack(rows).astype(np.float32)
    return (np.concatenate([X, X_syn]),
            np.concatenate([y, np.ones(len(X_syn), dtype=y.dtype)]))

# ── CTGAN oversampling on score space ─────────────────────────────────────────
def ctgan_oversample(X, y, seed=0):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    min_idx = np.where(y == 1)[0]
    maj_idx = np.where(y == 0)[0]
    n_needed = len(maj_idx) - len(min_idx)
    if n_needed <= 0 or len(min_idx) < 5:
        return X, y
    # Fit CTGAN on minority class feature vectors only
    cols = [f"f{i}" for i in range(X.shape[1])]
    df_min = pd.DataFrame(X[min_idx], columns=cols)
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_min)
    cg = CTGANSynthesizer(meta, epochs=50, verbose=False)
    cg.fit(df_min)
    synth = cg.sample(num_rows=n_needed).replace([np.inf, -np.inf], np.nan).dropna()
    X_syn = synth[cols[:X.shape[1]]].to_numpy(np.float32)
    if len(X_syn) == 0:
        return X, y
    return (np.concatenate([X, X_syn]),
            np.concatenate([y, np.ones(len(X_syn), dtype=y.dtype)]))

# ── Triage policy ─────────────────────────────────────────────────────────────
def apply_triage(probs, sig_max, pol):
    out = []
    for p, s in zip(probs, sig_max):
        if p >= pol["tau_high"]:
            out.append("AUTO_CONTAIN")
        elif p >= pol["tau"] and s >= pol["kappa"]:
            out.append("ESCALATE")
        else:
            out.append("ALLOW")
    return np.array(out)

# ═════════════════════════════════════════════════════════════════════════════
# Shared setup: load all models, build training + test score features
# ═════════════════════════════════════════════════════════════════════════════
print("Loading fixed models …")
know_m, know_sc = load_knowledge()
sem_m           = load_semantic()
know_lookup     = build_knowledge_lookup(know_m, know_sc)

_blob   = json.loads((CKPT / "decision_fuser_3agent.json").read_text())
_sc_ref = StandardScaler()
_sc_ref.mean_  = np.array(_blob["scaler_mean"],  dtype=np.float32)
_sc_ref.scale_ = np.array(_blob["scaler_scale"], dtype=np.float32)

print("  Building semantic fill …")
sem_fill = build_semantic_fill(sem_m, _sc_ref)
print(f"  Semantic fill: p_s={sem_fill[0]:.4f}  σ_s={sem_fill[1]:.2e}")

# Build 6D training score features, out-of-fold (no stacking leak)
print("\nBuilding 6D training score features (out-of-fold) …")
X_tr_score, y_tr, fam_tr = build_oof_train_score_features(know_m, know_lookup, sem_fill)
print(f"  X_tr_score: {X_tr_score.shape}  ransomware prev={y_tr.mean():.3f}")

# Build 6D test score features for all 4 splits
print("\nBuilding test score features for all 4 splits …")
X_te: dict   = {}
y_te: dict   = {}
fam_te: dict = {}
for sp in SPLIT_NAMES:
    print(f"  {sp} …")
    beh_m_sp, beh_sc_sp = load_behavioral_canonical(sp)
    X_beh_sp, y_sp, fam_sp = load_ransmap(sp, "test")
    X_te[sp]  = score_features(X_beh_sp, fam_sp, beh_m_sp, beh_sc_sp,
                                know_m, know_lookup, sem_fill)
    y_te[sp]  = y_sp
    fam_te[sp] = fam_sp
    print(f"    n={len(y_sp)}  RANSOM={y_sp.mean():.3f}")

# Load triage policy for E-G
policy = json.loads((CKPT / "policy_auto.json").read_text())
print(f"\nPolicy: tau={policy['tau']}  kappa={policy['kappa']}  tau_high={policy['tau_high']}")

results: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# E-E  Stronger fusion baselines
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-E  Stronger Fusion Baselines")
print("="*65)

# Mean Fusion: (p_b + p_k) / 2 from post-PC columns (cols 1 and 2)
def mean_fusion_probs(X_feat):
    return (X_feat[:, 1] + X_feat[:, 2]) / 2.0

# Train Decision MLP-1L (seed=0, no aug) once on X_tr_score
print("\n  Training Decision MLP-1L …")
mlp1l, sc1l = train_mlp1l(X_tr_score, y_tr.astype(np.float32), seed=0)

# Train Decision MLP-2L (seed=0, no aug)
print("  Training Decision MLP-2L …")
mlp2l, sc2l = train_mlp2l(X_tr_score, y_tr.astype(np.float32), seed=0)

# Train RF and GBT
print("  Training Random Forest …")
sc_sklearn = StandardScaler().fit(X_tr_score)
X_tr_scaled = sc_sklearn.transform(X_tr_score)
rf_clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
rf_clf.fit(X_tr_scaled, y_tr)

print("  Training Gradient Boosting (XGBoost substitute) …")
gbt_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
gbt_clf.fit(X_tr_scaled, y_tr)

ee_results = {}
for sp in SPLIT_NAMES:
    Xte = X_te[sp]; yte = y_te[sp]
    Xte_scaled = sc_sklearn.transform(Xte)

    probs_mf   = mean_fusion_probs(Xte)
    probs_mlp1 = predict_mlp(mlp1l, sc1l, Xte)
    probs_mlp2 = predict_mlp(mlp2l, sc2l, Xte)
    probs_rf   = rf_clf.predict_proba(Xte_scaled)[:, 1]
    probs_gbt  = gbt_clf.predict_proba(Xte_scaled)[:, 1]

    def _row(name, probs):
        return {
            "auroc": round(float(roc_auc_score(yte, probs)), 4),
            "auprc": round(float(average_precision_score(yte, probs)), 4),
            "ece":   round(ece_score(yte, probs), 4),
        }

    ee_results[sp] = {
        "BK_mean":   _row("BK_mean", probs_mf),
        "MLP_1L":    _row("MLP_1L", probs_mlp1),
        "MLP_2L":    _row("MLP_2L", probs_mlp2),
        "RF":        _row("RF", probs_rf),
        "GBT":       _row("GBT", probs_gbt),
    }
    print(f"  {sp}:")
    for mname, r in ee_results[sp].items():
        print(f"    {mname:12s}  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  ECE={r['ece']:.4f}")

results["E_E"] = ee_results

# Figure E-E: AUROC per method × split
methods  = ["BK_mean", "RF", "GBT", "MLP_2L", "MLP_1L"]
mlabels  = ["BK Mean", "Random Forest", "Grad. Boosting", "MLP-2L", "MLP-1L (ours)"]
colors_e = ["#999999", "#fdae61", "#f46d43", "#74add1", "#d73027"]
x = np.arange(len(SPLIT_NAMES)); w = 0.15

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

ax = axes[0]
for i, (m, lbl, col) in enumerate(zip(methods, mlabels, colors_e)):
    vals = [ee_results[sp][m]["auroc"] for sp in SPLIT_NAMES]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=lbl, color=col, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(SPLIT_LABELS)
ax.set_ylabel("AUROC")
ax.set_ylim(0.5, 1.12)
ax.set_title("Fusion Method AUROC Across Evaluation Splits", fontsize=9, fontweight="bold")
ax.legend(fontsize=6.5, loc="lower left", ncol=2)

ax2 = axes[1]
for i, (m, lbl, col) in enumerate(zip(methods, mlabels, colors_e)):
    vals = [ee_results[sp][m]["ece"] for sp in SPLIT_NAMES]
    bars = ax2.bar(x + (i - 2) * w, vals, w, label=lbl, color=col, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=5.5, rotation=90)
ax2.set_xticks(x); ax2.set_xticklabels(SPLIT_LABELS)
ax2.set_ylabel("ECE (lower is better)")
ax2.set_ylim(0, 0.35)
ax2.set_title("Calibration Error (ECE) Across Evaluation Splits", fontsize=9, fontweight="bold")
ax2.legend(fontsize=6.5, loc="upper left", ncol=2)

plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_ee_baselines.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_ee_baselines.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\n  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# E-F  CTGAN ablation under severe class imbalance
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-F  CTGAN Ablation Under Severe Imbalance")
print("="*65)

# Subsampling to target ransomware prevalence (keep all benign, subsample ransom)
PREVALENCES = [0.10, 0.05, 0.01]

rng_ef = np.random.default_rng(42)
idx_benign = np.where(y_tr == 0)[0]
idx_ransom = np.where(y_tr == 1)[0]
n_benign   = len(idx_benign)
print(f"  Full training set: {len(y_tr)} samples  ({n_benign} benign, {len(idx_ransom)} ransomware)")

ef_results = {}
Xte_rs = X_te["random_stratified"]
yte_rs = y_te["random_stratified"]

for prev in PREVALENCES:
    n_ransom_sub = max(5, int(round(n_benign * prev / (1 - prev))))
    n_ransom_sub = min(n_ransom_sub, len(idx_ransom))
    sub_ransom   = rng_ef.choice(idx_ransom, size=n_ransom_sub, replace=False)
    idx_sub      = np.concatenate([idx_benign, sub_ransom])
    X_sub = X_tr_score[idx_sub].astype(np.float32)
    y_sub = y_tr[idx_sub].astype(np.float32)
    actual_prev = float(y_sub.mean())
    print(f"\n  Prevalence target={prev:.0%}  actual={actual_prev:.3f}  "
          f"n={len(y_sub)} ({n_ransom_sub} ransom, {n_benign} benign)")

    ef_results[f"prev_{int(prev*100)}pct"] = {}

    for aug_name in ["none", "smote", "ctgan"]:
        print(f"    aug={aug_name} …", end=" ", flush=True)
        X_aug, y_aug = X_sub.copy(), y_sub.copy()
        if aug_name == "smote":
            X_aug, y_aug = smote_oversample(X_aug, y_aug, seed=0)
        elif aug_name == "ctgan":
            try:
                X_aug, y_aug = ctgan_oversample(X_aug, y_aug, seed=0)
            except Exception as e:
                print(f"[CTGAN failed: {e}] falling back to SMOTE …", end=" ")
                X_aug, y_aug = smote_oversample(X_aug, y_aug, seed=0)

        net_f, sc_f = train_mlp1l(X_aug, y_aug, seed=0)
        probs_f = predict_mlp(net_f, sc_f, Xte_rs)
        row = {
            "auroc": round(float(roc_auc_score(yte_rs, probs_f)), 4),
            "auprc": round(float(average_precision_score(yte_rs, probs_f)), 4),
            "ece":   round(ece_score(yte_rs, probs_f), 4),
            "n_train": int(len(y_aug)),
            "actual_prev": round(actual_prev, 4),
        }
        ef_results[f"prev_{int(prev*100)}pct"][aug_name] = row
        print(f"AUROC={row['auroc']:.4f}  AUPRC={row['auprc']:.4f}  ECE={row['ece']:.4f}")

results["E_F"] = ef_results

# Figure E-F: heatmap AUROC (prevalence × augmentation) + AUPRC
augs   = ["none", "smote", "ctgan"]
aug_labels = ["No Aug", "SMOTE", "CTGAN"]
prev_keys  = [f"prev_{int(p*100)}pct" for p in PREVALENCES]
prev_labels = ["10%", "5%", "1%"]

auroc_mat = np.array([[ef_results[pk][a]["auroc"] for a in augs] for pk in prev_keys])
auprc_mat = np.array([[ef_results[pk][a]["auprc"] for a in augs] for pk in prev_keys])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, mat, title, vmin in [
    (axes[0], auroc_mat, "AUROC", 0.5),
    (axes[1], auprc_mat, "AUPRC", 0.3),
]:
    im = ax.imshow(mat, vmin=vmin, vmax=1.0, cmap="RdYlGn", aspect="auto")
    for i in range(len(PREVALENCES)):
        for j in range(len(augs)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(augs))); ax.set_xticklabels(aug_labels)
    ax.set_yticks(range(len(PREVALENCES))); ax.set_yticklabels(prev_labels)
    ax.set_xlabel("Augmentation Method")
    ax.set_ylabel("Ransomware Prevalence")
    ax.set_title(f"Decision MLP {title}\n(test: random-stratified)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_ef_ctgan_ablation.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_ef_ctgan_ablation.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\n  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# E-G  Repeated leave-one-family-out
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  E-G  Leave-One-Family-Out")
print("="*65)

# Raw (unscored) random_stratified test windows — re-scored per LOFO
# iteration below with that iteration's own family-excluded checkpoint.
X_beh_te_rs, yte_rs, famte_rs = load_ransmap("random_stratified", "test")

eg_results = {}

for fam_held in RANSOM_FAMILIES:
    print(f"\n  Holding out: {fam_held} …")

    # Out-of-fold Decision-MLP training features from the family-excluded pool.
    # splits/lofo_exclude_{fam_held} already has fam_held's sessions removed
    # entirely; splits/lofo_exclude_{fam_held}_fold_{k} 5-fold cross-fits
    # *within* that reduced pool so these training features are genuinely
    # out-of-fold (no stacking leak), not just family-excluded.
    feats_g, ys_g = [], []
    for k in range(5):
        beh_m_k, beh_sc_k = load_behavioral_canonical(f"lofo_exclude_{fam_held}_fold_{k}")
        X_k, y_k, fam_k = load_ransmap(f"lofo_exclude_{fam_held}_fold_{k}", "test")
        feats_g.append(score_features(X_k, fam_k, beh_m_k, beh_sc_k,
                                       know_m, know_lookup, sem_fill))
        ys_g.append(y_k)
    X_tr_lofo = np.concatenate(feats_g)
    y_tr_lofo = np.concatenate(ys_g).astype(np.float32)
    n_rem = int((fam_tr == fam_held).sum())
    print(f"    LOFO train (out-of-fold): {len(y_tr_lofo)} windows across 5 folds "
          f"(ransom prev={y_tr_lofo.mean():.3f}); {fam_held} windows excluded "
          f"from main train: {n_rem}")

    # Retrain Decision MLP-1L
    net_g, sc_g = train_mlp1l(X_tr_lofo, y_tr_lofo, seed=0)

    # Test-time scoring uses the checkpoint trained on the FULL
    # family-excluded pool (never saw fam_held at all) — using the canonical
    # random_stratified checkpoint here would leak the Behavioural agent's own
    # exposure to the "held-out" family into a supposedly zero-shot result.
    beh_m_g, beh_sc_g = load_behavioral_canonical(f"lofo_exclude_{fam_held}")
    Xte_g = score_features(X_beh_te_rs, famte_rs, beh_m_g, beh_sc_g,
                            know_m, know_lookup, sem_fill)
    sig_max_te_g = np.maximum(Xte_g[:, 4], Xte_g[:, 5])  # max(sig_b, sig_k)

    probs_g = predict_mlp(net_g, sc_g, Xte_g)

    # Overall AUROC on all test windows
    overall_auroc = float(roc_auc_score(yte_rs, probs_g))

    # Mask for held-out family in test set (all ransomware)
    fam_mask = famte_rs == fam_held
    n_fam_te = int(fam_mask.sum())
    if n_fam_te == 0:
        print(f"    [{fam_held}] not in RS test set — skipping per-family metrics")
        eg_results[fam_held] = {
            "overall_auroc": round(overall_auroc, 4),
            "family_detection_rate": None,
            "family_escalation_rate": None,
            "n_family_test_windows": 0,
        }
        continue

    probs_fam  = probs_g[fam_mask]
    sig_max_fam = sig_max_te_g[fam_mask]

    # Detection rate: fraction of held-out family windows classified as ransomware
    det_rate = float((probs_fam >= 0.5).mean())

    # Escalation rate: fraction routed to ESCALATE by triage policy
    decs = apply_triage(probs_fam, sig_max_fam, policy)
    esc_rate = float((decs == "ESCALATE").mean())
    ac_rate  = float((decs == "AUTO_CONTAIN").mean())

    eg_results[fam_held] = {
        "overall_auroc":         round(overall_auroc, 4),
        "family_detection_rate": round(det_rate, 4),
        "family_escalation_rate": round(esc_rate, 4),
        "family_autocontain_rate": round(ac_rate, 4),
        "n_family_test_windows": n_fam_te,
        "n_train_removed":       n_rem,
    }
    print(f"    Overall AUROC={overall_auroc:.4f}  "
          f"FamilyDetect={det_rate:.4f}  "
          f"Escalation={esc_rate:.4f}  AutoContain={ac_rate:.4f}  "
          f"n_te={n_fam_te}")

results["E_G"] = eg_results

# Figure E-G: per-family overall AUROC + detection rate + escalation rate
fams_with_data = [f for f in RANSOM_FAMILIES if eg_results[f]["family_detection_rate"] is not None]
aurocs_eg = [eg_results[f]["overall_auroc"] for f in fams_with_data]
det_rates = [eg_results[f]["family_detection_rate"] for f in fams_with_data]
esc_rates = [eg_results[f]["family_escalation_rate"] for f in fams_with_data]
ac_rates  = [eg_results[f]["family_autocontain_rate"] for f in fams_with_data]

x_eg = np.arange(len(fams_with_data))
w_eg = 0.28
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

ax_l = axes[0]
bars = ax_l.bar(x_eg, aurocs_eg, 0.55, color="#2c7bb6", alpha=0.85, edgecolor="white")
for bar, v in zip(bars, aurocs_eg):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
              ha="center", va="bottom", fontsize=7)
ax_l.set_xticks(x_eg); ax_l.set_xticklabels(fams_with_data, rotation=20, ha="right")
ax_l.set_ylabel("AUROC (all test windows)")
ax_l.set_ylim(0.5, 1.10)
ax_l.set_title("Overall AUROC When Family\nExcluded from Training", fontsize=9, fontweight="bold")
ax_l.axhline(0.9755, color="gray", ls="--", lw=1.0, label="Full-train MLP ref")
ax_l.legend(fontsize=7)

ax_r = axes[1]
ax_r.bar(x_eg - w_eg, det_rates, w_eg, label="Detection Rate (≥0.5)", color="#1a9641", alpha=0.85)
ax_r.bar(x_eg,        ac_rates,  w_eg, label="Auto-Contain",          color="#d7191c", alpha=0.85)
ax_r.bar(x_eg + w_eg, esc_rates, w_eg, label="Escalation",            color="#fdae61", alpha=0.85)
for i, (d, ac, e) in enumerate(zip(det_rates, ac_rates, esc_rates)):
    ax_r.text(i - w_eg, d + 0.015, f"{d:.2f}", ha="center", fontsize=6.5)
    ax_r.text(i,        ac + 0.015, f"{ac:.2f}", ha="center", fontsize=6.5)
    ax_r.text(i + w_eg, e + 0.015, f"{e:.2f}", ha="center", fontsize=6.5)
ax_r.set_xticks(x_eg); ax_r.set_xticklabels(fams_with_data, rotation=20, ha="right")
ax_r.set_ylabel("Rate (held-out family test windows)")
ax_r.set_ylim(0, 1.20)
ax_r.set_title("Triage Decisions for Unseen Family\n(held-out from MLP training)", fontsize=9, fontweight="bold")
ax_r.legend(fontsize=7, loc="upper right")

plt.tight_layout(pad=0.6)
out = FIG_DIR / "fig_eg_lofo.pdf"
plt.savefig(out, dpi=DPI, bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_eg_lofo.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\n  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Save results JSON
# ═════════════════════════════════════════════════════════════════════════════
out_json = RES_DIR / "ee_eg_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {out_json}")
print("Done.")
