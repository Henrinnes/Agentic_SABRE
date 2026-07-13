#!/usr/bin/env python3
"""
eval_sabre2.py — SABRE-II evaluation for TIFS3 paper tables.

Design note
-----------
RanSMAP (behavioral), RDset (semantic), and RansKnow (knowledge) are
INDEPENDENT datasets — no single executable appears in all three, so no
RanSMAP window has a matched Semantic embedding or Knowledge transcript.

All RanSMAP evaluations below therefore use the full six-dimensional score
pool [S+B+K] (Semantic, Behavioural, Knowledge), NOT [B+K] as earlier
comments in this file used to say: Semantic and Knowledge each contribute a
fixed, label-independent corpus-mean score (their own MC Dropout mean/std
over their own corpus, RDset and RansKnow respectively), identical for
every RanSMAP sample regardless of family, computed once by
build_semantic_fill() and build_knowledge_lookup(). Only the Behavioural
Agent carries genuine per-sample signal in this evaluation. Knowledge is
NOT looked up by class_name/true family label -- an earlier version of
this lookup was and is a label leak, since it lets the RanSMAP sample's own
true family identity reach the Decision MLP; build_knowledge_lookup() now
ignores its `cn` argument entirely and always returns the corpus mean.

The Decision MLP's training-feature construction scores the Behavioural
Agent out-of-fold (5-fold session-grouped cross-fitting over
random_stratified's train sessions, see build_oof_behavioural_scores()) --
scoring training windows with a checkpoint trained on those same windows
would be stacking leakage. Test-time scoring uses the full-training-set
checkpoint on genuinely held-out test data, which requires no
cross-fitting.

See Remark 1 ("Cross-Dataset Fusion Without Per-Sample Pairing") in
TIFS3_Submission/Main.tex for the paper-facing version of this note.

Experiments
-----------
  E1  AUROC/AUPRC: [B] and [S+B+K] across 4 splits (Table II)
  E2  Calibration ECE: [B+K] mean fusion vs [S+B+K] Decision MLP (Table III)
  E3  CTGAN ablation on the [S+B+K] Decision MLP: no-aug / SMOTE / CTGAN (Table IV)
  E4  Peer Consultation: family holdout per-family recovery (Table IV, Supplementary)
  E5  Policy: fixed τ vs grid vs Bayesian TPE, on [S+B+K] Decision MLP output (Table V)
  E6  Latency: [S+B+K] Decision MLP overhead vs [B+K] mean fusion (Table VI)
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as scipy_t

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
CKPT    = ROOT / "checkpoints"
SPLITS  = ROOT / "splits"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RANSMAP  = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
RDSET    = "/home/henry_k/Datasets/RDset/semantic_dataset_with_embeddings.parquet"
RANSKNOW = "/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_307.csv"

BEHAVIOR_COLS = [
    "avg_read_throughput","avg_write_throughput",
    "read_LBA_variance","write_LBA_variance",
    "write_entropy_mean","mem_read_entropy_mean","mem_write_entropy_mean",
    "mem_readwrite_entropy_mean","mem_read_GPA_variance",
    "mem_write_GPA_variance","mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance","mem_read_count","mem_write_count",
    "mem_readwrite_count","mem_exec_count",
]

T_MC    = 10
BATCH   = 512
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG     = np.random.default_rng(42)
K_SEEDS = 6
META   = json.loads((CKPT / "metadata.json").read_text())

# ── CNN definitions matching saved checkpoints ────────────────────────────────
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

class CNNSemantic(nn.Module):
    def __init__(self, in_dim=384, p=0.3):
        super().__init__()
        pool_dim = (in_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1,32,3,padding="same"), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p), nn.Flatten(),
            nn.Linear(pool_dim*32,128), nn.ReLU(),
            nn.Linear(128,1), nn.Sigmoid(),
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
    def __init__(self, in_dim=4, hidden=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden,1), nn.Sigmoid(),
        )
    def forward(self,x): return self.net(x).squeeze(-1)

def _enable_mc(m):
    if isinstance(m, nn.Dropout): m.train()

def mc_infer(model, X: np.ndarray, T=T_MC) -> tuple[np.ndarray, np.ndarray]:
    model.to(DEVICE)
    model.apply(_enable_mc)
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    preds = []
    with torch.no_grad():
        for _ in range(T): preds.append(model(t).cpu().numpy())
    preds = np.stack(preds)
    return preds.mean(0), preds.std(0)

def mc_infer_batched(model, X: np.ndarray, T=T_MC):
    means, stds = [], []
    for i in range(0, len(X), BATCH):
        m, s = mc_infer(model, X[i:i+BATCH], T)
        means.append(m); stds.append(s)
    return np.concatenate(means), np.concatenate(stds)

# ── loaders ───────────────────────────────────────────────────────────────────
def load_behavioral(split: str):
    dim = META["behavioral_input_dim"]
    m   = CNNBehavioral(dim)
    m.load_state_dict(torch.load(str(CKPT/f"behavioral_cnn_{split}.pt"), map_location="cpu"))
    m.eval()
    sc  = joblib.load(str(CKPT/f"behavioral_scaler_{split}.joblib"))
    return m, sc

def load_semantic():
    dim = META["semantic_input_dim"]
    m   = CNNSemantic(dim)
    m.load_state_dict(torch.load(str(CKPT/"semantic_cnn.pt"), map_location="cpu"), strict=True)
    m.eval()
    return m

def load_knowledge():
    dim = META["knowledge_input_dim"]
    m   = CNNKnowledge(dim)
    with torch.no_grad(): _ = m(torch.zeros(1,1,dim))
    m.load_state_dict(torch.load(str(CKPT/"knowledge_cnn.pt"), map_location="cpu"))
    m.eval()
    sc  = joblib.load(str(CKPT/"knowledge_scaler.joblib"))
    return m, sc

def load_decision_mlp():
    blob   = json.loads((CKPT/"decision_fuser_3agent.json").read_text())
    in_dim = blob["in_dim"]
    cfg    = blob["cfg"]
    mlp    = FusionMLP(in_dim, cfg["hidden"], cfg["p_drop"])
    state  = {k: torch.tensor(v, dtype=torch.float32)
              for k, v in blob["model_state"].items()}
    mlp.load_state_dict(state, strict=True)
    mlp.eval().to(DEVICE)
    sc = StandardScaler()
    sc.mean_  = np.array(blob["scaler_mean"],  dtype=np.float32)
    sc.scale_ = np.array(blob["scaler_scale"], dtype=np.float32)
    T_cal = float(np.exp(blob.get("temp_logT", 0.0)))
    return mlp, sc, T_cal, in_dim

def load_beh_split(split: str, subset: str):
    sess = set((SPLITS/split/f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[BEHAVIOR_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values if "class_name" in sub.columns \
           else np.array(["unknown"]*len(sub))
    return X, y, fam

def build_oof_behavioural_scores(n_folds=5):
    """Out-of-fold Behavioural MC-Dropout scores for random_stratified's full
    train set, used to build the Decision MLP's training features.

    Scoring the training data with the same checkpoint trained on it (in-sample)
    is stacking leakage: the MLP would learn from an optimistically confident
    base-agent score it will never see again at inference. splits/cv_fold_{k}
    partition random_stratified's train sessions into 5 disjoint session-level
    folds (generate_cv_lofo_splits.py); checkpoints/behavioral_cnn_cv_fold_{k}.pt
    is trained on the other 4 folds only, so scoring fold k's held-out sessions
    with it is a genuine out-of-fold prediction. Concatenating all 5 folds
    covers the entire train set without any sample being scored by a model that
    trained on it.
    """
    y_all, fam_all, p_all, sig_all = [], [], [], []
    for k in range(n_folds):
        beh_m_k, beh_sc_k = load_behavioral(f"cv_fold_{k}")
        X_k, y_k, fam_k   = load_beh_split(f"cv_fold_{k}", "test")
        Xs_k              = beh_sc_k.transform(X_k).astype(np.float32)
        p_k, sig_k        = mc_infer_batched(beh_m_k, Xs_k)
        y_all.append(y_k); fam_all.append(fam_k)
        p_all.append(p_k); sig_all.append(sig_k)
    return (np.concatenate(p_all), np.concatenate(sig_all),
            np.concatenate(y_all).astype(int), np.concatenate(fam_all))

def build_knowledge_lookup(km, ksc):
    """Return a label-independent corpus-mean Knowledge vector for every RanSMAP
    sample, regardless of its family.

    RansKnow and RanSMAP share no real per-sample pairing (a RanSMAP execution
    window has no corresponding threat-intelligence transcript). A per-family
    lookup keyed by the sample's own true family label would leak the label
    being predicted into the Knowledge Agent's input, so we deliberately do not
    condition on `cn` at all — this mirrors how the Semantic Agent's
    `build_semantic_fill` is already a single label-independent constant.
    """
    df  = pd.read_csv(RANSKNOW)
    fc  = META["knowledge_features"]
    Xrk = ksc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    corpus_mean = Xrk.mean(0)
    def lookup(cn: str):
        return corpus_mean
    return lookup

def build_semantic_fill(sem_m) -> tuple[float, float]:
    """Return a label-independent (p_s_fill, sig_s_fill) from the corpus-mean embedding.

    RDset has no family column so we cannot do a per-family lookup.
    Instead we compute the mean embedding across all 138 k PE files and run
    it through the Semantic CNN with MC-Dropout — giving a single neutral
    semantic score that is in-distribution for the pretrained 6-D MLP scaler
    (which saw σ_s ≈ 2.75e-8, i.e. near-zero uncertainty).
    """
    df  = pd.read_parquet(RDSET)
    emb = np.stack(df["embedding"].values).mean(axis=0, keepdims=True).astype(np.float32)
    p_s, sig_s = mc_infer(sem_m, emb, T=T_MC)
    return float(p_s[0]), float(sig_s[0])

# ── metrics ───────────────────────────────────────────────────────────────────
def ece(y, probs, n_bins=15) -> float:
    frac, mean_p = calibration_curve(y, probs, n_bins=n_bins, strategy="uniform")
    counts  = np.histogram(probs, bins=n_bins, range=(0,1))[0]
    weights = counts / (counts.sum() + 1e-9)
    return float(np.sum(weights[:len(frac)] * np.abs(frac - mean_p)))

# ── peer consultation ─────────────────────────────────────────────────────────
KL  = 0.15
EPS = 1e-6

def peer_consult(p_s, p_b, p_k, sig_s, sig_b, sig_k, kl=KL):
    p_s, p_b, p_k = p_s.copy(), p_b.copy(), p_k.copy()
    for mask, own_s, peers in [
        (sig_s > kl, sig_s, [(p_b, sig_b), (p_k, sig_k)]),
        (sig_b > kl, sig_b, [(p_s, sig_s), (p_k, sig_k)]),
        (sig_k > kl, sig_k, [(p_s, sig_s), (p_b, sig_b)]),
    ]:
        if mask.any():
            ws = [1/(s[mask]+EPS) for _, s in peers]
            ws_sum = sum(ws)
            adj = sum(w * p[mask] for w, (p, _) in zip(ws, peers)) / ws_sum
            if own_s is sig_s: p_s[mask] = adj
            elif own_s is sig_b: p_b[mask] = adj
            else: p_k[mask] = adj
    return p_s, p_b, p_k

# ── fusion helpers ────────────────────────────────────────────────────────────
def fuse_mean_bk(p_b, p_k): return (p_b + p_k) / 2.0
def fuse_mean_3(p_s, p_b, p_k): return (p_s + p_b + p_k) / 3.0

def fuse_mlp(mlp, sc, T_cal, feat_vec: np.ndarray) -> np.ndarray:
    Xsc = sc.transform(feat_vec).astype(np.float32)
    with torch.no_grad():
        probs = mlp(torch.from_numpy(Xsc).to(DEVICE)).cpu().numpy()
    if T_cal != 1.0:
        eps = 1e-6; p = np.clip(probs, eps, 1-eps)
        logits = np.log(p) - np.log(1-p)
        probs  = 1 / (1 + np.exp(-logits / T_cal))
    return probs

def apply_triage(probs, sig_max, pol):
    tau, kappa = pol["tau"], pol["kappa"]
    tau_h, kl  = pol["tau_high"], pol["kappa_low"]
    decisions  = []
    for p, s in zip(probs, sig_max):
        if p >= tau_h and s <= kl:    decisions.append("AUTO_CONTAIN")
        elif p >= tau or s > kappa:   decisions.append("ESCALATE")
        else:                         decisions.append("ALLOW")
    return decisions

# ── score a behavioural split using B+K with corpus-mean semantic fill ────────
def score_bk(split, beh_m, beh_sc, know_m, know_sc, know_lookup,
             mlp, mlp_sc, T_cal, sem_fill: tuple[float, float], use_pc=True):
    """Evaluate on a behavioral split.

    Semantic scores come from the corpus-mean semantic fill (label-independent,
    in-distribution for the pretrained 6-D MLP scaler).  Peer Consultation still
    runs over all three agents; semantic peer-adjust is a no-op since σ_s ≈ 0 < KL.
    """
    p_s_fill, sig_s_fill = sem_fill
    X_beh, y, fam = load_beh_split(split, "test")
    Xs = beh_sc.transform(X_beh).astype(np.float32)
    p_b, sig_b = mc_infer_batched(beh_m, Xs)

    Xk = np.stack([know_lookup(f) for f in fam]).astype(np.float32)
    p_k, sig_k = mc_infer_batched(know_m, Xk)

    n = len(p_b)
    p_s   = np.full(n, p_s_fill,   dtype=np.float32)
    sig_s = np.full(n, sig_s_fill, dtype=np.float32)

    if use_pc:
        p_s, p_b, p_k = peer_consult(p_s, p_b, p_k, sig_s, sig_b, sig_k)

    probs_mean = fuse_mean_bk(p_b, p_k)

    # Full 6-D feature vector matching the pretrained MLP training distribution
    feat = np.stack([p_s, p_b, p_k, sig_s, sig_b, sig_k], axis=1).astype(np.float32)
    probs_mlp = fuse_mlp(mlp, mlp_sc, T_cal, feat)

    sig_max = np.stack([sig_b, sig_k]).max(0)
    return {"y": y, "fam": fam,
            "p_b": p_b, "p_k": p_k, "sig_b": sig_b, "sig_k": sig_k,
            "p_s": p_s, "sig_s": sig_s,
            "probs_mean": probs_mean, "probs_mlp": probs_mlp,
            "sig_max": sig_max}


# ═════════════════════════════════════════════════════════════════════════════
def train_mlp_variant(X_tr_in, y_tr_in, X_te_in, y_te_in, aug="none", seed=42):
    """Train a FusionMLP with aug ∈ {none, smote, ctgan}.

    Returns (auroc, auprc, ece, trained_net, fitted_scaler).
    """
    rng = np.random.default_rng(seed)
    X_a, y_a = X_tr_in.copy(), y_tr_in.copy()

    if aug == "smote":
        from sklearn.neighbors import NearestNeighbors
        min_idx = np.where(y_a == 1)[0]
        target  = int((y_a==0).sum() - (y_a==1).sum())
        if target > 0 and len(min_idx) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(5, len(min_idx)-1)).fit(X_a[min_idx])
            _, idxs = nbrs.kneighbors(X_a[min_idx])
            new_rows = []
            for _ in range(target):
                i  = int(rng.integers(len(min_idx)))
                nb = int(idxs[i, rng.integers(1, idxs.shape[1])])
                lam = float(rng.random())
                new_rows.append(X_a[min_idx[i]]*lam + X_a[min_idx[nb]]*(1-lam))
            X_syn = np.stack(new_rows).astype(np.float32)
            X_a   = np.concatenate([X_a, X_syn])
            y_a   = np.concatenate([y_a, np.ones(len(X_syn), dtype=np.float32)])
    elif aug == "ctgan":
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            cols = [f"x{i}" for i in range(X_a.shape[1])]
            df_  = pd.DataFrame(X_a, columns=cols)
            df_["label"] = y_a.astype(int)
            meta_ = SingleTableMetadata()
            meta_.detect_from_dataframe(df_)
            cg = CTGANSynthesizer(meta_, epochs=100, verbose=False)
            cg.fit(df_)
            synth = cg.sample(num_rows=2000).replace([np.inf,-np.inf], np.nan).dropna()
            X_a = np.concatenate([X_a, synth[cols].to_numpy(np.float32)])
            y_a = np.concatenate([y_a, synth["label"].to_numpy(np.float32)])
        except Exception as e_ctgan:
            print(f"    CTGAN failed ({e_ctgan}) — falling back to SMOTE")
            min_idx = np.where(y_a == 1)[0]
            target  = int((y_a==0).sum() - (y_a==1).sum())
            if target > 0 and len(min_idx) > 1:
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(5, len(min_idx)-1)).fit(X_a[min_idx])
                _, idxs = nbrs.kneighbors(X_a[min_idx])
                new_rows = []
                for _ in range(target):
                    i  = int(rng.integers(len(min_idx)))
                    nb = int(idxs[i, rng.integers(1, idxs.shape[1])])
                    lam = float(rng.random())
                    new_rows.append(X_a[min_idx[i]]*lam + X_a[min_idx[nb]]*(1-lam))
                X_syn = np.stack(new_rows).astype(np.float32)
                X_a   = np.concatenate([X_a, X_syn])
                y_a   = np.concatenate([y_a, np.ones(len(X_syn), dtype=np.float32)])

    sc2   = StandardScaler().fit(X_a)
    X_as  = sc2.transform(X_a).astype(np.float32)
    X_tes = sc2.transform(X_te_in).astype(np.float32)

    X_t, X_v, y_t, y_v = train_test_split(
        X_as, y_a, test_size=0.2, stratify=(y_a>0.5).astype(int), random_state=seed)

    torch.manual_seed(seed)
    net = FusionMLP(X_a.shape[1], 32, 0.2).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce_loss = nn.BCELoss()
    loader   = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_t), torch.tensor(y_t)),
        batch_size=128, shuffle=True)

    best_va, best_state, pat = float("inf"), None, 0
    for _ in range(40):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = bce_loss(net(torch.tensor(X_v).to(DEVICE)),
                          torch.tensor(y_v).to(DEVICE)).item()
        if va + 1e-6 < best_va:
            best_va = va; pat = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 6: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        probs_out = net(torch.tensor(X_tes).to(DEVICE)).cpu().numpy()
    return (roc_auc_score(y_te_in, probs_out),
            average_precision_score(y_te_in, probs_out),
            ece(y_te_in, probs_out),
            net, sc2)


def main():
    print("="*70)
    print(f"SABRE-II Evaluation (TIFS2) — {K_SEEDS} seeds")
    print("="*70)

    print("\nLoading models …")
    sem_m                = load_semantic()
    know_m, know_sc      = load_knowledge()
    _, mlp_sc_ref, _, _  = load_decision_mlp()
    policy               = json.loads((CKPT/"policy_auto.json").read_text())
    know_lookup          = build_knowledge_lookup(know_m, know_sc)
    print("  Building corpus-mean semantic fill …")
    sem_fill             = build_semantic_fill(sem_m)
    sem_fill = (sem_fill[0], float(mlp_sc_ref.mean_[3]))
    print(f"  Semantic fill: p_s={sem_fill[0]:.4f}  σ_s={sem_fill[1]:.2e}")
    print(f"  Policy (TPE): τ={policy['tau']:.4f}  κ={policy['kappa']:.4f}  "
          f"τ_h={policy['tau_high']:.4f}  κ_l={policy['kappa_low']:.4f}")

    splits = ["random_stratified","family_holdout","hardware_holdout","temporal_holdout"]

    # ── Preload behavioural models for all splits ─────────────────────────
    print("\nPre-loading behavioural models …")
    beh_models = {}
    for s in splits:
        beh_models[s] = load_behavioral(s)
    beh_m_rs, beh_sc_rs = beh_models["random_stratified"]

    # ── Build 6-D MLP training score space (out-of-fold, no stacking leak) ─
    print("\nBuilding 6-D training score space (random_stratified, out-of-fold) …")
    p_b_tr, sig_b_tr, y_tr, fam_tr = build_oof_behavioural_scores()
    Xk_tr    = np.stack([know_lookup(f) for f in fam_tr]).astype(np.float32)
    p_k_tr, sig_k_tr = mc_infer_batched(know_m, Xk_tr)
    n_tr     = len(p_b_tr)
    p_s_tr   = np.full(n_tr, sem_fill[0], np.float32)
    sig_s_tr = np.full(n_tr, sem_fill[1], np.float32)
    p_s_tr_pc, p_b_tr_pc, p_k_tr_pc = peer_consult(
        p_s_tr, p_b_tr, p_k_tr, sig_s_tr, sig_b_tr, sig_k_tr)
    X_tr_score = np.stack([p_s_tr_pc, p_b_tr_pc, p_k_tr_pc,
                           sig_s_tr, sig_b_tr, sig_k_tr], axis=1).astype(np.float32)
    y_tr_score = y_tr.astype(np.float32)

    # ── Precompute test-set score matrices for all 4 splits (deterministic) ─
    print("\nPrecomputing test-set score matrices …")
    split_data = {}
    for split in splits:
        beh_m_s, beh_sc_s = beh_models[split]
        X_beh_s, y_s, fam_s = load_beh_split(split, "test")
        Xs_s     = beh_sc_s.transform(X_beh_s).astype(np.float32)
        p_b_s, sig_b_s = mc_infer_batched(beh_m_s, Xs_s)
        Xk_s     = np.stack([know_lookup(f) for f in fam_s]).astype(np.float32)
        p_k_s, sig_k_s = mc_infer_batched(know_m, Xk_s)
        n_s      = len(p_b_s)
        p_s_s    = np.full(n_s, sem_fill[0], np.float32)
        sig_s_s  = np.full(n_s, sem_fill[1], np.float32)
        # With-PC features (used for BK_mlp E1 and E4 pc-on)
        p_s_pc, p_b_pc, p_k_pc = peer_consult(
            p_s_s.copy(), p_b_s.copy(), p_k_s.copy(),
            sig_s_s.copy(), sig_b_s.copy(), sig_k_s.copy())
        X_te_pc  = np.stack([p_s_pc, p_b_pc, p_k_pc,
                              sig_s_s, sig_b_s, sig_k_s], axis=1).astype(np.float32)
        # Without-PC features (used for E4 pc-off baseline)
        X_te_npc = np.stack([p_s_s, p_b_s, p_k_s,
                              sig_s_s, sig_b_s, sig_k_s], axis=1).astype(np.float32)
        pm_s     = fuse_mean_bk(p_b_pc, p_k_pc)
        smax_s   = np.stack([sig_b_s, sig_k_s]).max(0)

        auroc_b    = round(roc_auc_score(y_s, p_b_s), 4)
        auprc_b    = round(average_precision_score(y_s, p_b_s), 4)
        auroc_mean = round(roc_auc_score(y_s, pm_s), 4)
        auprc_mean = round(average_precision_score(y_s, pm_s), 4)
        print(f"  [{split}]  n={len(y_s)}  ransom={int(y_s.sum())}  "
              f"B AUROC={auroc_b:.4f}  BK_mean AUROC={auroc_mean:.4f}")
        split_data[split] = {
            "y": y_s, "fam": fam_s, "p_b": p_b_s,
            "n_test":   int(len(y_s)),
            "n_ransom": int(y_s.sum()),
            "B":       {"auroc": auroc_b,    "auprc": auprc_b},
            "BK_mean": {"auroc": auroc_mean, "auprc": auprc_mean},
            "X_te_pc":  X_te_pc,
            "X_te_npc": X_te_npc,
            "sig_max":  smax_s,
        }

    # Aliases for the primary split used in E2/E3/E5
    y_te_rs    = split_data["random_stratified"]["y"]
    X_te_score = split_data["random_stratified"]["X_te_pc"]
    sig_max_rs = split_data["random_stratified"]["sig_max"]
    p_b_te     = split_data["random_stratified"]["p_b"]
    # probs_mean = mean of PC-adjusted p_b, p_k (columns 1,2 of X_te_pc)
    probs_mean_rs = fuse_mean_bk(X_te_score[:, 1], X_te_score[:, 2])

    # ── E2 deterministic parts ────────────────────────────────────────────
    T_sp  = 0.702
    eps_c = 1e-6
    logits_m  = (np.log(np.clip(probs_mean_rs, eps_c, 1-eps_c)) -
                 np.log(np.clip(1-probs_mean_rs, eps_c, 1-eps_c)))
    pm_ts         = 1 / (1 + np.exp(-logits_m / T_sp))
    ece_b         = round(ece(y_te_rs, p_b_te), 4)
    ece_mean_pre  = round(ece(y_te_rs, probs_mean_rs), 4)
    ece_mean_post = round(ece(y_te_rs, pm_ts), 4)

    # ── E4 deterministic B_dr per family ─────────────────────────────────
    fam_fh  = split_data["family_holdout"]["fam"]
    y_fh    = split_data["family_holdout"]["y"]
    p_b_fh  = split_data["family_holdout"]["p_b"]
    det_b_dr_fh = {}
    for f in sorted(set(fam_fh)):
        mask = (fam_fh == f)
        yf   = y_fh[mask]
        if yf.sum() == 0: continue
        ran  = yf == 1
        det_b_dr_fh[f] = {
            "n": int(mask.sum()), "ransom": int(ran.sum()),
            "B_dr": round(float((p_b_fh[mask][ran] >= 0.5).mean()), 4),
        }

    # ── Policies ─────────────────────────────────────────────────────────
    policies = {
        "sabre_plus_fixed": {"tau":0.10,"kappa":0.10,"tau_high":0.95,"kappa_low":0.08},
        "grid_search":      {"tau":0.40,"kappa":0.20,"tau_high":0.85,"kappa_low":0.10},
        "bayesian_tpe":     policy,
    }

    # ── Seed loop ─────────────────────────────────────────────────────────
    seed_results = []
    final_mlp_pair = None  # (eval_mlp, eval_sc) from last seed — used for E6

    for seed in range(K_SEEDS):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{K_SEEDS}  (seed={seed})")
        print(f"{'='*60}")

        # E3: CTGAN Ablation — train 3 MLP variants
        print("  E3: training MLP variants …")
        r_none  = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "none",  seed)
        r_smote = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "smote", seed)
        r_ctgan = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "ctgan", seed)
        print(f"    none  AUROC={r_none[0]:.4f}  AUPRC={r_none[1]:.4f}  ECE={r_none[2]:.4f}")
        print(f"    smote AUROC={r_smote[0]:.4f}  AUPRC={r_smote[1]:.4f}  ECE={r_smote[2]:.4f}")
        print(f"    ctgan AUROC={r_ctgan[0]:.4f}  AUPRC={r_ctgan[1]:.4f}  ECE={r_ctgan[2]:.4f}")

        eval_mlp, eval_sc = r_ctgan[3], r_ctgan[4]
        eval_mlp.eval()
        final_mlp_pair = (eval_mlp, eval_sc)

        # E1: MLP AUROC/AUPRC for all 4 splits
        print("  E1: scoring 4 splits …")
        e1_mlp = {}
        for split in splits:
            probs_s = fuse_mlp(eval_mlp, eval_sc, 1.0, split_data[split]["X_te_pc"])
            y_s     = split_data[split]["y"]
            e1_mlp[split] = (
                round(roc_auc_score(y_s, probs_s), 4),
                round(average_precision_score(y_s, probs_s), 4),
            )
            print(f"    [{split}]  AUROC={e1_mlp[split][0]:.4f}  AUPRC={e1_mlp[split][1]:.4f}")

        # E2: ECE for Decision MLP
        probs_mlp_rs = fuse_mlp(eval_mlp, eval_sc, 1.0, X_te_score)
        ece_mlp = round(ece(y_te_rs, probs_mlp_rs), 4)
        print(f"  E2: ECE_mlp={ece_mlp:.4f}")

        # E4: Peer Consultation — family holdout
        print("  E4: Peer Consultation …")
        probs_fh_npc = fuse_mlp(eval_mlp, eval_sc, 1.0, split_data["family_holdout"]["X_te_npc"])
        probs_fh_pc  = fuse_mlp(eval_mlp, eval_sc, 1.0, split_data["family_holdout"]["X_te_pc"])
        auroc_npc = round(roc_auc_score(y_fh, probs_fh_npc), 4)
        auroc_pc  = round(roc_auc_score(y_fh, probs_fh_pc),  4)
        e4_fam = {}
        for f, dets in det_b_dr_fh.items():
            mask = (fam_fh == f)
            ran  = y_fh[mask] == 1
            e4_fam[f] = (
                round(float((probs_fh_npc[mask][ran] >= 0.5).mean()), 4),
                round(float((probs_fh_pc[mask][ran]  >= 0.5).mean()), 4),
            )
        print(f"    AUROC no-PC={auroc_npc:.4f}  with-PC={auroc_pc:.4f}")

        # E5: Policy Comparison
        print("  E5: Policy Comparison …")
        e5 = {}
        for pname, pol in policies.items():
            decs  = apply_triage(probs_mlp_rs, sig_max_rs, pol)
            preds = np.array([0 if d=="ALLOW" else 1 for d in decs])
            ba    = round(balanced_accuracy_score(y_te_rs, preds), 4)
            esc   = round(decs.count("ESCALATE")     / len(decs), 4)
            ac    = round(decs.count("AUTO_CONTAIN")  / len(decs), 4)
            alw   = round(decs.count("ALLOW")         / len(decs), 4)
            e5[pname] = (ba, esc, ac, alw)
            print(f"    {pname:22s}  bal_acc={ba:.4f}  esc={esc:.3f}  ac={ac:.3f}  allow={alw:.3f}")

        seed_results.append({
            "e3": {
                "no_aug": (r_none[0],  r_none[1],  r_none[2]),
                "smote":  (r_smote[0], r_smote[1], r_smote[2]),
                "ctgan":  (r_ctgan[0], r_ctgan[1], r_ctgan[2]),
            },
            "e1_mlp":    e1_mlp,
            "e2_mlp_ece": ece_mlp,
            "e4": {"auroc_npc": auroc_npc, "auroc_pc": auroc_pc, "fam": e4_fam},
            "e5": e5,
        })

    # ── Aggregate across seeds with 95% t-CI ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"Aggregating {K_SEEDS} seeds …")

    def ci95(vals):
        a  = np.array(vals, dtype=float)
        n  = len(a)
        m  = float(a.mean())
        se = float(a.std(ddof=1)) / np.sqrt(n)
        lo, hi = scipy_t.interval(0.95, df=n-1, loc=m, scale=se)
        return {"mean": round(m,4), "ci95_lo": round(float(lo),4), "ci95_hi": round(float(hi),4)}

    out = {
        "K_seeds": K_SEEDS, "seeds": list(range(K_SEEDS)),
        "E1_main":{}, "E2_calibration":{}, "E3_ctgan_ablation":{},
        "E4_peer_consultation":{}, "E5_policy_comparison":{}, "E6_latency":{},
    }

    # E1
    for split in splits:
        out["E1_main"][split] = {
            **{k: split_data[split][k] for k in ("n_test","n_ransom","B","BK_mean")},
            "BK_mlp": {
                "auroc": ci95([sr["e1_mlp"][split][0] for sr in seed_results]),
                "auprc": ci95([sr["e1_mlp"][split][1] for sr in seed_results]),
                # Raw per-seed AUROC (paired with the fixed, single-checkpoint
                # Behavioural AUROC in split_data[split]["B"]["auroc"] for a
                # genuine one-sample Wilcoxon on the MLP's own training
                # randomness against the deployed B checkpoint).
                "auroc_seeds": [round(sr["e1_mlp"][split][0], 4) for sr in seed_results],
            },
        }

    # E2
    out["E2_calibration"] = {
        "B_ece":        ece_b,
        "BK_mean_pre":  ece_mean_pre,
        "BK_mean_post": ece_mean_post,
        "T_sp":         T_sp,
        "BK_mlp_ece":   ci95([sr["e2_mlp_ece"] for sr in seed_results]),
    }

    # E3
    for aug_key in ["no_aug","smote","ctgan"]:
        out["E3_ctgan_ablation"][aug_key] = {
            "auroc": ci95([sr["e3"][aug_key][0] for sr in seed_results]),
            "auprc": ci95([sr["e3"][aug_key][1] for sr in seed_results]),
            "ece":   ci95([sr["e3"][aug_key][2] for sr in seed_results]),
        }

    # E4
    out["E4_peer_consultation"] = {
        "auroc_no_pc":   ci95([sr["e4"]["auroc_npc"] for sr in seed_results]),
        "auroc_with_pc": ci95([sr["e4"]["auroc_pc"]  for sr in seed_results]),
        "per_family": {
            f: {
                **dets,
                "mlp_nopc": ci95([sr["e4"]["fam"][f][0] for sr in seed_results]),
                "mlp_pc":   ci95([sr["e4"]["fam"][f][1] for sr in seed_results]),
            }
            for f, dets in det_b_dr_fh.items()
        },
    }

    # E5
    for pname in policies:
        out["E5_policy_comparison"][pname] = {
            "balanced_accuracy": ci95([sr["e5"][pname][0] for sr in seed_results]),
            "escalation_rate":   ci95([sr["e5"][pname][1] for sr in seed_results]),
            "ac_rate":           ci95([sr["e5"][pname][2] for sr in seed_results]),
            "allow_rate":        ci95([sr["e5"][pname][3] for sr in seed_results]),
        }

    # ── E6: Latency (single-run with final seed's MLP) ────────────────────
    print("\n" + "─"*60)
    print("E6: Latency measurement  (batch=32, 500 reps)")
    print("─"*60)
    eval_mlp, eval_sc = final_mlp_pair
    X_lat, _, fam_lat = load_beh_split("random_stratified","test")
    Xs_lat   = beh_sc_rs.transform(X_lat[:256]).astype(np.float32)
    Xk_lat   = np.stack([know_lookup(f) for f in fam_lat[:256]]).astype(np.float32)
    n_lat    = 32
    ps_lat   = np.full(n_lat, sem_fill[0], np.float32)
    sigs_lat = np.full(n_lat, sem_fill[1], np.float32)

    for _ in range(30):
        mc_infer(beh_m_rs, Xs_lat[:32]); mc_infer(know_m, Xk_lat[:32])

    N_reps = 500
    t0 = time.perf_counter()
    for _ in range(N_reps):
        i = int(RNG.integers(0, 224))
        pb, sb = mc_infer(beh_m_rs, Xs_lat[i:i+32])
        pk, sk = mc_infer(know_m,   Xk_lat[i:i+32])
        _ = fuse_mean_bk(pb, pk)
    t_mean = (time.perf_counter()-t0)/N_reps*1000

    t0 = time.perf_counter()
    for _ in range(N_reps):
        i = int(RNG.integers(0, 224))
        pb, sb = mc_infer(beh_m_rs, Xs_lat[i:i+32])
        pk, sk = mc_infer(know_m,   Xk_lat[i:i+32])
        ps_c = ps_lat.copy(); sigs_c = sigs_lat.copy()
        ps_c, pb, pk = peer_consult(ps_c, pb, pk, sigs_c, sb, sk)
        feat_l = np.stack([ps_c, pb, pk, sigs_c, sb, sk], axis=1).astype(np.float32)
        _ = fuse_mlp(eval_mlp, eval_sc, 1.0, feat_l)
    t_mlp = (time.perf_counter()-t0)/N_reps*1000

    print(f"  SABRE+ mean fusion :  {t_mean:.3f} ms/batch-32")
    print(f"  SABRE-II MLP+PC    :  {t_mlp:.3f} ms/batch-32")
    print(f"  Overhead           : +{t_mlp-t_mean:.3f} ms  ({(t_mlp-t_mean)/t_mean*100:.1f}%)")
    out["E6_latency"] = {
        "batch_size": 32, "n_reps": N_reps,
        "sabre_plus_ms": round(t_mean, 3),
        "sabre2_ms":     round(t_mlp, 3),
        "overhead_ms":   round(t_mlp-t_mean, 3),
        "overhead_pct":  round((t_mlp-t_mean)/t_mean*100, 1),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = RESULTS / "sabre2_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{'='*70}")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    import os; os.chdir(ROOT)
    main()
