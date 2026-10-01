#!/usr/bin/env python3
"""
run_cost_sensitive_policy.py — R1#8: cost-sensitive triage objective and
full confusion-matrix breakdown of triage outcomes.

Reviewer 1 pointed out that balanced accuracy is a binary metric applied to
a three-action policy (AUTO_CONTAIN / ESCALATE / ALLOW) without stating how
the three actions collapse to a binary prediction, and asked for:
  (a) an explicit per-outcome confusion breakdown (ransomware incorrectly
      allowed; benign incorrectly auto-contained; ransomware escalated;
      benign escalated; auto-contain rate; auto-allow rate; analyst
      workload), and
  (b) a cost-sensitive objective reflecting that a missed ransomware
      (ALLOW on a true positive) is operationally far worse than a false
      auto-contain (AUTO_CONTAIN on a true negative), which is in turn
      worse than an unnecessary escalation (analyst time).

We define, for thresholds Theta = (tau_high, tau, kappa, kappa_low):
  J_cost(Theta) = -( w_fn * P(ransomware ALLOWed)
                    + w_fp * P(benign AUTO_CONTAINed)
                    + w_esc * P(ESCALATE) )
with w_fn=10, w_fp=5, w_esc=1 (missing ransomware costs 2x a false
auto-contain, which costs 5x an escalation) as an illustrative, clearly
documented weighting -- not a claim that these are the "correct" costs for
every deployment.

Uses the same out-of-fold Behavioural + corpus-mean Knowledge score-pool
construction as the rest of this codebase (no stacking leak), and a freshly
trained Decision MLP (seed=0) scored on a held-out validation split for
threshold search and the genuinely held-out test split for final reporting.

Outputs: results/cost_sensitive_policy.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as _tts
from Agentic_SABRE.knowledge_features import RANSKNOW_CSV

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_MC   = 10
META   = json.loads((CKPT / "metadata.json").read_text())
W_FN, W_FP, W_ESC = 10.0, 5.0, 1.0  # illustrative operational costs

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

def load_behavioral(split):
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

def load_beh_split(split, subset):
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
    return np.concatenate(p_all), np.concatenate(sig_all), np.concatenate(y_all).astype(int)

def corpus_mean_knowledge(know_m, know_sc):
    df  = pd.read_csv(RANSKNOW)
    fc  = META["knowledge_features"]
    Xrk = know_sc.transform(df[fc].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
    corp = Xrk.mean(0)
    p_k, sig_k = mc_infer_batched(know_m, corp[None, :])
    return float(p_k[0]), float(sig_k[0])

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
        return net(torch.tensor(Xs).to(DEVICE)).cpu().numpy().ravel()

def apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low):
    decisions = np.full(len(p_hat), "ALLOW", dtype=object)
    auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
    decisions[auto_mask] = "AUTO_CONTAIN"
    esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
    decisions[esc_mask] = "ESCALATE"
    return decisions

def confusion_breakdown(decisions, y):
    y = np.asarray(y, dtype=int)
    n = len(y)
    ransom_allowed      = int(((decisions == "ALLOW") & (y == 1)).sum())
    benign_contained    = int(((decisions == "AUTO_CONTAIN") & (y == 0)).sum())
    ransom_escalated    = int(((decisions == "ESCALATE") & (y == 1)).sum())
    benign_escalated    = int(((decisions == "ESCALATE") & (y == 0)).sum())
    ransom_contained_ok = int(((decisions == "AUTO_CONTAIN") & (y == 1)).sum())
    benign_allowed_ok   = int(((decisions == "ALLOW") & (y == 0)).sum())
    return {
        "n": n,
        "ransomware_incorrectly_allowed":   ransom_allowed,
        "ransomware_incorrectly_allowed_rate": round(ransom_allowed / n, 4),
        "benign_incorrectly_autocontained": benign_contained,
        "benign_incorrectly_autocontained_rate": round(benign_contained / n, 4),
        "ransomware_escalated":  ransom_escalated,
        "ransomware_escalated_rate": round(ransom_escalated / n, 4),
        "benign_escalated":      benign_escalated,
        "benign_escalated_rate": round(benign_escalated / n, 4),
        "ransomware_correctly_autocontained": ransom_contained_ok,
        "benign_correctly_allowed": benign_allowed_ok,
        "auto_contain_rate": round(float((decisions == "AUTO_CONTAIN").mean()), 4),
        "escalate_rate":     round(float((decisions == "ESCALATE").mean()), 4),
        "allow_rate":        round(float((decisions == "ALLOW").mean()), 4),
        "analyst_workload_rate": round(float((decisions == "ESCALATE").mean()), 4),
    }

def cost_objective(decisions, y):
    y = np.asarray(y, dtype=int)
    n = len(y)
    p_fn  = ((decisions == "ALLOW") & (y == 1)).sum() / n
    p_fp  = ((decisions == "AUTO_CONTAIN") & (y == 0)).sum() / n
    p_esc = (decisions == "ESCALATE").mean()
    return -(W_FN * p_fn + W_FP * p_fp + W_ESC * p_esc)

def balanced_acc_objective(decisions, y, esc_penalty=0.3):
    alarm = (decisions != "ALLOW").astype(int)
    return balanced_accuracy_score(y, alarm) - esc_penalty * (decisions == "ESCALATE").mean()

print("Loading models …")
know_m, know_sc = load_knowledge()
p_know, sig_know = corpus_mean_knowledge(know_m, know_sc)

print("Building out-of-fold Behavioural training scores …")
p_b_tr, sig_b_tr, y_tr = build_oof_behavioural_scores()
n_tr = len(p_b_tr)
p_s_tr = np.full(n_tr, 0.5, np.float32)   # neutral Semantic placeholder (constant, non-informative)
sig_s_tr = np.full(n_tr, 0.1, np.float32)
p_k_tr = np.full(n_tr, p_know, np.float32)
sig_k_tr = np.full(n_tr, sig_know, np.float32)
X_tr = np.stack([p_s_tr, p_b_tr, p_k_tr, sig_s_tr, sig_b_tr, sig_k_tr], axis=1).astype(np.float32)

print("Training fresh Decision MLP (seed=0) on out-of-fold features …")
mlp, mlp_sc = train_fusion_mlp(X_tr, y_tr, in_dim=6, seed=0)

def score_split(split, subset):
    beh_m, beh_sc = load_behavioral(split)
    X_beh, y = load_beh_split(split, subset)
    Xs = beh_sc.transform(X_beh).astype(np.float32)
    p_b, sig_b = mc_infer_batched(beh_m, Xs)
    n = len(p_b)
    feat = np.stack([np.full(n,0.5,np.float32), p_b, np.full(n,p_know,np.float32),
                      np.full(n,0.1,np.float32), sig_b, np.full(n,sig_know,np.float32)], axis=1).astype(np.float32)
    probs = predict(mlp, mlp_sc, feat)
    sig_max = np.maximum(sig_b, sig_know)
    return probs, sig_max, y

print("Scoring val split (threshold search) and test split (final report) …")
p_val, sig_val, y_val   = score_split("random_stratified", "val")
p_test, sig_test, y_test = score_split("random_stratified", "test")
print(f"  val n={len(y_val)}  test n={len(y_test)}")

results: dict = {}

# ═════════════════════════════════════════════════════════════════════════════
# Fit two policies on val, evaluate both on test: balanced-accuracy (existing
# style) vs cost-sensitive (new)
# ═════════════════════════════════════════════════════════════════════════════
def run_optuna(objective_fn, n_trials=150):
    def obj(trial):
        tau_high  = trial.suggest_float("tau_high", 0.5, 0.99)
        tau       = trial.suggest_float("tau", 0.05, tau_high)
        kappa     = trial.suggest_float("kappa", 0.01, 0.5)
        kappa_low = trial.suggest_float("kappa_low", 0.01, kappa)
        decisions = apply_policy(p_val, sig_val, tau, kappa, tau_high, kappa_low)
        return objective_fn(decisions, y_val)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

print("\nFitting balanced-accuracy policy (existing objective, for comparison) …")
ba_params = run_optuna(balanced_acc_objective)
print(f"  {ba_params}")

print("Fitting cost-sensitive policy (new objective: w_fn=10, w_fp=5, w_esc=1) …")
cost_params = run_optuna(cost_objective)
print(f"  {cost_params}")

for name, params in [("balanced_accuracy_policy", ba_params), ("cost_sensitive_policy", cost_params)]:
    decisions_test = apply_policy(p_test, sig_test, params["tau"], params["kappa"],
                                   params["tau_high"], params["kappa_low"])
    breakdown = confusion_breakdown(decisions_test, y_test)
    breakdown["thresholds"] = params
    breakdown["balanced_accuracy"] = round(float(balanced_accuracy_score(
        y_test, (decisions_test != "ALLOW").astype(int))), 4)
    breakdown["cost_objective_value"] = round(float(cost_objective(decisions_test, y_test)), 4)
    results[name] = breakdown
    print(f"\n  {name}:")
    for k, v in breakdown.items():
        print(f"    {k}: {v}")

results["cost_weights"] = {"w_fn_ransomware_allowed": W_FN, "w_fp_benign_autocontained": W_FP, "w_esc_escalation": W_ESC}

out_path = RESULTS / "cost_sensitive_policy.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out_path}")
print("Done.")
