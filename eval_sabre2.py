#!/usr/bin/env python3
"""
eval_sabre2.py — SABRE-II evaluation for TIFS2 paper tables.

Design note
-----------
RanSMAP (behavioral), RDset (semantic), and RansKnow (knowledge) are
INDEPENDENT datasets — no single executable appears in all three.
Therefore the main [B+K] evaluation is done on the behavioral test split
with knowledge looked up by class_name (same approach as TIFS1).
Semantic scores are included in a separate family-level lookup experiment.

Experiments
-----------
  E1  AUROC/AUPRC: [B] and [B+K] across 4 splits (Table I)
  E2  Calibration ECE: mean fusion vs Decision MLP (Table II)
  E3  CTGAN ablation: no-aug / SMOTE / CTGAN (Table III)
  E4  Peer Consultation: family holdout per-family recovery (Table IV)
  E5  Policy: fixed τ vs grid vs Bayesian TPE (Table V)
  E6  Latency: Decision MLP overhead vs mean fusion (Table VI)
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

T_MC   = 10
BATCH  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG    = np.random.default_rng(42)
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

def build_knowledge_lookup(km, ksc):
    df  = pd.read_csv(RANSKNOW)
    fc  = META["knowledge_features"]
    Xrk = ksc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    fm  = {}
    for i, row in df.iterrows():
        for f in [x.strip() for x in str(row.get("Family_List","")).split(",")]:
            if f and f != "nan":
                fm.setdefault(f, []).append(Xrk[i])
    fm          = {f: np.mean(np.stack(v), 0) for f, v in fm.items()}
    corpus_mean = Xrk.mean(0)
    def lookup(cn: str):
        if cn in fm: return fm[cn]
        for k in fm:
            if k.lower() in cn.lower() or cn.lower() in k.lower():
                return fm[k]
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
def train_mlp_variant(X_tr_in, y_tr_in, X_te_in, y_te_in, aug="none"):
    """Train a FusionMLP with aug ∈ {none, smote, ctgan}.

    Returns (auroc, auprc, ece, trained_net, fitted_scaler).
    """
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
                i  = int(RNG.integers(len(min_idx)))
                nb = int(idxs[i, RNG.integers(1, idxs.shape[1])])
                lam = float(RNG.random())
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
                    i  = int(RNG.integers(len(min_idx)))
                    nb = int(idxs[i, RNG.integers(1, idxs.shape[1])])
                    lam = float(RNG.random())
                    new_rows.append(X_a[min_idx[i]]*lam + X_a[min_idx[nb]]*(1-lam))
                X_syn = np.stack(new_rows).astype(np.float32)
                X_a   = np.concatenate([X_a, X_syn])
                y_a   = np.concatenate([y_a, np.ones(len(X_syn), dtype=np.float32)])

    sc2   = StandardScaler().fit(X_a)
    X_as  = sc2.transform(X_a).astype(np.float32)
    X_tes = sc2.transform(X_te_in).astype(np.float32)

    X_t, X_v, y_t, y_v = train_test_split(
        X_as, y_a, test_size=0.2, stratify=(y_a>0.5).astype(int), random_state=42)

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
    print("SABRE-II Evaluation (TIFS2)")
    print("="*70)

    print("\nLoading models …")
    sem_m              = load_semantic()
    know_m, know_sc    = load_knowledge()
    _, mlp_sc_ref, _, _ = load_decision_mlp()   # used only for σ_s clamp
    policy             = json.loads((CKPT/"policy_auto.json").read_text())
    know_lookup        = build_knowledge_lookup(know_m, know_sc)
    print("  Building corpus-mean semantic fill (label-independent) …")
    sem_fill           = build_semantic_fill(sem_m)
    # σ_s from MC-Dropout on any embedding is out-of-distribution for the
    # 6-D score scaler (training mean σ_s ≈ 2.75e-08).  Clamp to that mean
    # so the scaled σ_s lands at 0 and does not saturate hidden neurons.
    sem_fill = (sem_fill[0], float(mlp_sc_ref.mean_[3]))
    print(f"  Semantic fill: p_s={sem_fill[0]:.4f}  σ_s={sem_fill[1]:.2e}")
    print(f"  Policy (TPE): τ={policy['tau']:.4f}  κ={policy['kappa']:.4f}  "
          f"τ_h={policy['tau_high']:.4f}  κ_l={policy['kappa_low']:.4f}")

    out = {"E1_main":{}, "E2_calibration":{},
           "E3_ctgan_ablation":{}, "E4_peer_consultation":{},
           "E5_policy_comparison":{}, "E6_latency":{}}

    # ── Build random_stratified 6-D training/test score spaces ───────────
    print("\nBuilding 6-D score spaces (random_stratified) …")
    beh_m_rs, beh_sc_rs = load_behavioral("random_stratified")

    X_beh_tr, y_tr, fam_tr = load_beh_split("random_stratified","train")
    Xs_tr = beh_sc_rs.transform(X_beh_tr).astype(np.float32)
    p_b_tr, sig_b_tr = mc_infer_batched(beh_m_rs, Xs_tr)
    Xk_tr = np.stack([know_lookup(f) for f in fam_tr]).astype(np.float32)
    p_k_tr, sig_k_tr = mc_infer_batched(know_m, Xk_tr)
    n_tr = len(p_b_tr)
    p_s_tr   = np.full(n_tr, sem_fill[0], dtype=np.float32)
    sig_s_tr = np.full(n_tr, sem_fill[1], dtype=np.float32)
    # Apply PC so training features match what score_bk produces at eval time
    p_s_tr_pc, p_b_tr_pc, p_k_tr_pc = peer_consult(p_s_tr, p_b_tr, p_k_tr,
                                                     sig_s_tr, sig_b_tr, sig_k_tr)
    X_tr_score = np.stack([p_s_tr_pc, p_b_tr_pc, p_k_tr_pc,
                           sig_s_tr, sig_b_tr, sig_k_tr], axis=1).astype(np.float32)
    y_tr_score = y_tr.astype(np.float32)

    # Build test score space with PC (consistent with score_bk)
    X_beh_te, y_te_rs, fam_te = load_beh_split("random_stratified","test")
    Xs_te = beh_sc_rs.transform(X_beh_te).astype(np.float32)
    p_b_te, sig_b_te = mc_infer_batched(beh_m_rs, Xs_te)
    Xk_te = np.stack([know_lookup(f) for f in fam_te]).astype(np.float32)
    p_k_te, sig_k_te = mc_infer_batched(know_m, Xk_te)
    n_te = len(p_b_te)
    p_s_te   = np.full(n_te, sem_fill[0], dtype=np.float32)
    sig_s_te = np.full(n_te, sem_fill[1], dtype=np.float32)
    p_s_te_pc, p_b_te_pc, p_k_te_pc = peer_consult(p_s_te, p_b_te, p_k_te,
                                                     sig_s_te, sig_b_te, sig_k_te)
    X_te_score = np.stack([p_s_te_pc, p_b_te_pc, p_k_te_pc,
                           sig_s_te, sig_b_te, sig_k_te], axis=1).astype(np.float32)
    probs_mean_rs = fuse_mean_bk(p_b_te_pc, p_k_te_pc)
    sig_max_rs    = np.stack([sig_b_te, sig_k_te]).max(0)

    # ── E3: CTGAN Ablation (train first — CTGAN MLP becomes primary) ──────
    print("\n" + "─"*60)
    print("E3: CTGAN Ablation — train Decision MLP variants (6-D score space)")
    print("─"*60)
    print("  No-aug …")
    r_none  = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "none")
    print("  SMOTE …")
    r_smote = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "smote")
    print("  CTGAN (100 epochs on 6-D score space) …")
    r_ctgan = train_mlp_variant(X_tr_score, y_tr_score, X_te_score, y_te_rs, "ctgan")

    for name, r in [("none",r_none),("smote",r_smote),("ctgan",r_ctgan)]:
        print(f"  {name:6s}  AUROC={r[0]:.4f}  AUPRC={r[1]:.4f}  ECE={r[2]:.4f}")
    out["E3_ctgan_ablation"] = {
        "no_aug": {"auroc":round(r_none[0],4),  "auprc":round(r_none[1],4),  "ece":round(r_none[2],4)},
        "smote":  {"auroc":round(r_smote[0],4), "auprc":round(r_smote[1],4), "ece":round(r_smote[2],4)},
        "ctgan":  {"auroc":round(r_ctgan[0],4), "auprc":round(r_ctgan[1],4), "ece":round(r_ctgan[2],4)},
    }

    # Primary eval MLP = CTGAN-augmented variant (trained on random_stratified)
    eval_mlp, eval_sc = r_ctgan[3], r_ctgan[4]
    eval_mlp.eval()

    # ── E1: Main AUROC/AUPRC across 4 splits ─────────────────────────────
    print("\n" + "─"*60)
    print("E1: Main AUROC / AUPRC across 4 splits")
    print("─"*60)

    splits = ["random_stratified","family_holdout","hardware_holdout","temporal_holdout"]
    for split in splits:
        beh_m_s, beh_sc_s = (beh_m_rs, beh_sc_rs) if split == "random_stratified" \
                             else load_behavioral(split)
        res = score_bk(split, beh_m_s, beh_sc_s, know_m, know_sc, know_lookup,
                       eval_mlp, eval_sc, 1.0, sem_fill, use_pc=True)
        y = res["y"]
        auroc_b    = roc_auc_score(y, res["p_b"])
        auprc_b    = average_precision_score(y, res["p_b"])
        auroc_mean = roc_auc_score(y, res["probs_mean"])
        auprc_mean = average_precision_score(y, res["probs_mean"])
        auroc_mlp  = roc_auc_score(y, res["probs_mlp"])
        auprc_mlp  = average_precision_score(y, res["probs_mlp"])
        print(f"\n  [{split}]  n={len(y)}  ransom={y.sum()}")
        print(f"    [B]             AUROC={auroc_b:.4f}  AUPRC={auprc_b:.4f}")
        print(f"    [B+K] mean      AUROC={auroc_mean:.4f}  AUPRC={auprc_mean:.4f}")
        print(f"    [B+K] DecisionMLP  AUROC={auroc_mlp:.4f}  AUPRC={auprc_mlp:.4f}")
        out["E1_main"][split] = {
            "n_test": int(len(y)), "n_ransom": int(y.sum()),
            "B":       {"auroc": round(auroc_b,4),    "auprc": round(auprc_b,4)},
            "BK_mean": {"auroc": round(auroc_mean,4), "auprc": round(auprc_mean,4)},
            "BK_mlp":  {"auroc": round(auroc_mlp,4),  "auprc": round(auprc_mlp,4)},
        }

    # ── E2: Calibration ECE ───────────────────────────────────────────────
    print("\n" + "─"*60)
    print("E2: Calibration ECE  (random_stratified)")
    print("─"*60)

    # SABRE+ temperature T=0.702 applied to mean fusion
    T_sp = 0.702
    eps_c = 1e-6
    logits_m = np.log(np.clip(probs_mean_rs,eps_c,1-eps_c)) - \
               np.log(np.clip(1-probs_mean_rs,eps_c,1-eps_c))
    pm_ts = 1 / (1 + np.exp(-logits_m / T_sp))

    # SABRE-II Decision MLP (CTGAN, T=1.0)
    X_te_sc2 = eval_sc.transform(X_te_score).astype(np.float32)
    with torch.no_grad():
        probs_mlp_rs = eval_mlp(torch.tensor(X_te_sc2).to(DEVICE)).cpu().numpy()

    ece_b         = ece(y_te_rs, p_b_te)
    ece_mean_pre  = ece(y_te_rs, probs_mean_rs)
    ece_mean_post = ece(y_te_rs, pm_ts)
    ece_mlp       = ece(y_te_rs, probs_mlp_rs)
    print(f"  [B]              ECE = {ece_b:.4f}")
    print(f"  [B+K] mean  pre-T = {ece_mean_pre:.4f}  post-T({T_sp}) = {ece_mean_post:.4f}")
    print(f"  [B+K] DecisionMLP ECE = {ece_mlp:.4f}")
    out["E2_calibration"] = {
        "B_ece": round(ece_b,4),
        "BK_mean_pre": round(ece_mean_pre,4), "BK_mean_post": round(ece_mean_post,4), "T_sp": T_sp,
        "BK_mlp_ece":  round(ece_mlp,4),
    }

    # ── E4: Peer Consultation — family holdout ────────────────────────────
    print("\n" + "─"*60)
    print("E4: Peer Consultation — family holdout per-family detection rate")
    print("─"*60)

    beh_m_fh, beh_sc_fh = load_behavioral("family_holdout")
    res_pc_off = score_bk("family_holdout", beh_m_fh, beh_sc_fh, know_m, know_sc,
                           know_lookup, eval_mlp, eval_sc, 1.0, sem_fill, use_pc=False)
    res_pc_on  = score_bk("family_holdout", beh_m_fh, beh_sc_fh, know_m, know_sc,
                           know_lookup, eval_mlp, eval_sc, 1.0, sem_fill, use_pc=True)
    y_fh  = res_pc_on["y"]
    fam_fh= res_pc_on["fam"]
    pc4   = {}

    auroc_npc = roc_auc_score(y_fh, res_pc_off["probs_mlp"])
    auroc_pc  = roc_auc_score(y_fh, res_pc_on["probs_mlp"])
    print(f"  Overall AUROC  no-PC={auroc_npc:.4f}  with-PC={auroc_pc:.4f}")

    for f in sorted(set(fam_fh)):
        mask = (fam_fh == f)
        yf   = y_fh[mask]
        if yf.sum() == 0: continue
        ran  = yf == 1
        dr_b   = float((res_pc_on["p_b"][mask][ran] >= 0.5).mean())
        dr_npc = float((res_pc_off["probs_mlp"][mask][ran] >= 0.5).mean())
        dr_pc  = float((res_pc_on["probs_mlp"][mask][ran] >= 0.5).mean())
        print(f"  {f:25s}  [B]={dr_b:.3f}  MLP_noPC={dr_npc:.3f}  MLP+PC={dr_pc:.3f}")
        pc4[f] = {"n":int(mask.sum()), "ransom":int(ran.sum()),
                  "B_dr":round(dr_b,4), "mlp_nopc":round(dr_npc,4), "mlp_pc":round(dr_pc,4)}
    out["E4_peer_consultation"] = {
        "auroc_no_pc": round(auroc_npc,4), "auroc_with_pc": round(auroc_pc,4),
        "per_family": pc4,
    }

    # ── E5: Policy Comparison ─────────────────────────────────────────────
    print("\n" + "─"*60)
    print("E5: Policy comparison  (random_stratified)")
    print("─"*60)

    policies = {
        "sabre_plus_fixed": {"tau":0.10,"kappa":0.10,"tau_high":0.95,"kappa_low":0.08},
        "grid_search":      {"tau":0.40,"kappa":0.20,"tau_high":0.85,"kappa_low":0.10},
        "bayesian_tpe":     policy,
    }
    for pname, pol in policies.items():
        decs  = apply_triage(probs_mlp_rs, sig_max_rs, pol)
        preds = np.array([0 if d=="ALLOW" else 1 for d in decs])
        ba    = balanced_accuracy_score(y_te_rs, preds)
        esc   = decs.count("ESCALATE") / len(decs)
        ac    = decs.count("AUTO_CONTAIN") / len(decs)
        alw   = decs.count("ALLOW") / len(decs)
        print(f"  {pname:22s}  bal_acc={ba:.4f}  esc={esc:.3f}  ac={ac:.3f}  allow={alw:.3f}")
        out["E5_policy_comparison"][pname] = {
            "balanced_accuracy": round(ba,4), "escalation_rate": round(esc,4),
            "ac_rate": round(ac,4), "allow_rate": round(alw,4),
        }

    # ── E6: Latency ───────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("E6: Latency measurement  (batch=32, 500 reps)")
    print("─"*60)

    X_lat, _, fam_lat = load_beh_split("random_stratified","test")
    Xs_lat  = beh_sc_rs.transform(X_lat[:256]).astype(np.float32)
    Xk_lat  = np.stack([know_lookup(f) for f in fam_lat[:256]]).astype(np.float32)
    n_lat   = 32
    ps_lat  = np.full(n_lat, sem_fill[0], dtype=np.float32)
    sigs_lat = np.full(n_lat, sem_fill[1], dtype=np.float32)

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
        "batch_size":32, "n_reps":N_reps,
        "sabre_plus_ms": round(t_mean,3), "sabre2_ms": round(t_mlp,3),
        "overhead_ms": round(t_mlp-t_mean,3),
        "overhead_pct": round((t_mlp-t_mean)/t_mean*100,1),
    }

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = RESULTS / "sabre2_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{'='*70}")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    import os; os.chdir(ROOT)
    main()
