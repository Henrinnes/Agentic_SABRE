#!/usr/bin/env python3
"""Rerun of eval_ood_variants.py, extended to also compute genuine per-variant
escalation rates under the real Bayesian-TPE policy (checkpoints/policy_auto.json).

Recall: BehavioralAgent (checkpoints/behavioral_cnn.pt, the deployed/production
checkpoint eval_ood_variants.py already used -- no Knowledge lookup involved
here, so this script never had the per-family leak bug the others did).

Escalation rate needs the fused (p_hat, sigma_max) the triage policy actually
operates on: Semantic/Knowledge corpus-mean fills (same recipe as
eval_sabre2.py) + a Decision MLP trained on out-of-fold random_stratified
features (same recipe as gen_triage_space_data.py), applied to this
BehavioralAgent's p_b/sig_b on the OOD variants set.

Saves results/ood_variants_eval.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).parent)
import eval_sabre2 as E
from Agentic_SABRE.agents.behavioural import BehavioralAgent

PARQUET   = os.path.join(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"), "ransmap_extracted/ransmap_ood_variants.parquet")
THRESHOLD = 0.5

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

df = pd.read_parquet(PARQUET)
print(f"OOD variants: {len(df):,} windows | variants: {sorted(df['class_name'].unique())}")

X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
y = df["label"].astype(int).values  # all 1s
variants = df["class_name"].values

agent = BehavioralAgent(T=10)
outputs = agent.score_batch([{"features": v} for v in X])
p_b  = np.array([o.prob   for o in outputs])
sig_b = np.array([o.uncert for o in outputs])
preds = (p_b >= THRESHOLD).astype(int)

print(f"Overall recall: {preds.mean():.4f}")

# ── Fused (p_hat, sigma_max) + real triage policy ───────────────────────────
policy = json.loads((E.CKPT / "policy_auto.json").read_text())
print("policy:", policy)

sem_m           = E.load_semantic()
know_m, know_sc = E.load_knowledge()
_, mlp_sc_ref, _, _ = E.load_decision_mlp()
know_lookup     = E.build_knowledge_lookup(know_m, know_sc)
sem_fill = E.build_semantic_fill(sem_m)
sem_fill = (sem_fill[0], float(mlp_sc_ref.mean_[3]))

print("Building OOF training score space + training CTGAN Decision MLP (seed=0) …")
p_b_tr, sig_b_tr, y_tr, fam_tr = E.build_oof_behavioural_scores()
Xk_tr = np.stack([know_lookup(f) for f in fam_tr]).astype(np.float32)
p_k_tr, sig_k_tr = E.mc_infer_batched(know_m, Xk_tr)
n_tr = len(p_b_tr)
p_s_tr   = np.full(n_tr, sem_fill[0], np.float32)
sig_s_tr = np.full(n_tr, sem_fill[1], np.float32)
p_s_tr_pc, p_b_tr_pc, p_k_tr_pc = E.peer_consult(
    p_s_tr, p_b_tr, p_k_tr, sig_s_tr, sig_b_tr, sig_k_tr)
X_tr_score = np.stack([p_s_tr_pc, p_b_tr_pc, p_k_tr_pc,
                        sig_s_tr, sig_b_tr, sig_k_tr], axis=1).astype(np.float32)
y_tr_score = y_tr.astype(np.float32)

beh_m_rs, beh_sc_rs = E.load_behavioral("random_stratified")
X_beh_rs, y_rs, fam_rs = E.load_beh_split("random_stratified", "test")
Xs_rs = beh_sc_rs.transform(X_beh_rs).astype(np.float32)
p_b_rs, sig_b_rs = E.mc_infer_batched(beh_m_rs, Xs_rs)
Xk_rs = np.stack([know_lookup(f) for f in fam_rs]).astype(np.float32)
p_k_rs, sig_k_rs = E.mc_infer_batched(know_m, Xk_rs)
n_rs = len(p_b_rs)
p_s_rs   = np.full(n_rs, sem_fill[0], np.float32)
sig_s_rs = np.full(n_rs, sem_fill[1], np.float32)
p_s_pc_rs, p_b_pc_rs, p_k_pc_rs = E.peer_consult(
    p_s_rs.copy(), p_b_rs.copy(), p_k_rs.copy(),
    sig_s_rs.copy(), sig_b_rs.copy(), sig_k_rs.copy())
X_te_score = np.stack([p_s_pc_rs, p_b_pc_rs, p_k_pc_rs,
                        sig_s_rs, sig_b_rs, sig_k_rs], axis=1).astype(np.float32)

auroc, auprc, ece_v, eval_mlp, eval_sc = E.train_mlp_variant(
    X_tr_score, y_tr_score, X_te_score, y_rs, aug="ctgan", seed=0)
eval_mlp.eval()
print(f"  sanity: random_stratified test AUROC={auroc:.4f}")

# ── Fuse the OOD variants set with this Decision MLP ────────────────────────
n_ood = len(p_b)
Xk_ood = np.stack([know_lookup("Conti") for _ in range(n_ood)]).astype(np.float32)
p_k_ood, sig_k_ood = E.mc_infer_batched(know_m, Xk_ood)
p_s_ood   = np.full(n_ood, sem_fill[0], np.float32)
sig_s_ood = np.full(n_ood, sem_fill[1], np.float32)
p_s_pc, p_b_pc, p_k_pc = E.peer_consult(
    p_s_ood.copy(), p_b.copy().astype(np.float32), p_k_ood.copy(),
    sig_s_ood.copy(), sig_b.copy().astype(np.float32), sig_k_ood.copy())
feat_ood = np.stack([p_s_pc, p_b_pc, p_k_pc, sig_s_ood, sig_b, sig_k_ood],
                     axis=1).astype(np.float32)
p_hat_ood = E.fuse_mlp(eval_mlp, eval_sc, 1.0, feat_ood)
sig_max_ood = np.maximum(sig_b, sig_k_ood)

decisions = np.array(E.apply_triage(p_hat_ood, sig_max_ood, policy))
esc_mask  = decisions == "ESCALATE"
ac_mask   = decisions == "AUTO_CONTAIN"

print(f"Overall: ESCALATE={esc_mask.mean():.4f}  AUTO_CONTAIN={ac_mask.mean():.4f}  "
      f"ALLOW={(decisions=='ALLOW').mean():.4f}")

out = {
    "n_windows": int(len(df)),
    "policy": policy,
    "threshold_naive": THRESHOLD,
    "overall": {
        "recall_behavioural": round(float(preds.mean()), 4),
        "mean_prob": round(float(p_b.mean()), 4),
        "mean_uncert": round(float(sig_b.mean()), 4),
        "esc_rate": round(float(esc_mask.mean()), 4),
        "auto_contain_rate": round(float(ac_mask.mean()), 4),
    },
    "per_variant": {},
}

for variant in sorted(df["class_name"].unique()):
    mask = variants == variant
    n = int(mask.sum())
    recall = float(preds[mask].mean())
    esc = float(esc_mask[mask].mean())
    ac  = float(ac_mask[mask].mean())
    out["per_variant"][variant] = {
        "n": n, "recall": round(recall, 4),
        "esc_rate": round(esc, 4), "auto_contain_rate": round(ac, 4),
        "mean_prob": round(float(p_b[mask].mean()), 4),
        "mean_uncert": round(float(sig_b[mask].mean()), 4),
    }
    print(f"  {variant:<12} n={n:5d}  recall={recall:.4f}  esc={esc:.4f}  ac={ac:.4f}")

out_path = E.RESULTS / "ood_variants_eval.json"
out_path.write_text(json.dumps(out, indent=2))
print("saved ->", out_path)
