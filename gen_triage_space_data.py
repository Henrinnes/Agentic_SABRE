#!/usr/bin/env python3
"""One-off: compute real (p_hat, sigma_max) scatter data for fig_triage_space.

Replicates eval_sabre2.py main()'s actual E1 training path: the Decision
MLP used for the paper's genuine AUROC numbers is trained fresh (CTGAN
variant, out-of-fold Behavioural features) inside main() -- the on-disk
checkpoints/decision_fuser_3agent.json is a stale pre-fix artifact never
retrained after the corpus-mean-always Knowledge/Semantic fix, and produces
saturated (~1.0) output when fed post-fix constant Semantic/Knowledge fills.
Using the same training recipe as main() avoids that mismatch.

Ransomware / Benign (train dist.) come from random_stratified's test split.
Unseen family (OOD) comes from family_holdout's test split, restricted to
the two families that split's Behavioural checkpoint never trained on
(REvil, WannaCry) -- scored with the same MLP, since main() applies its one
trained Decision MLP across all 4 splits' test sets.

Saves results/triage_space_scatter.json for generate_figures.py to plot.
"""
import json
import os
from pathlib import Path

import numpy as np

os.chdir(Path(__file__).parent)
import eval_sabre2 as E

SEED = 0

policy = json.loads((E.CKPT / "policy_auto.json").read_text())
print("policy:", policy)

sem_m           = E.load_semantic()
know_m, know_sc = E.load_knowledge()
_, mlp_sc_ref, _, _ = E.load_decision_mlp()
know_lookup     = E.build_knowledge_lookup(know_m, know_sc)

sem_fill = E.build_semantic_fill(sem_m)
sem_fill = (sem_fill[0], float(mlp_sc_ref.mean_[3]))
print(f"Semantic fill: p_s={sem_fill[0]:.4f}  sig_s={sem_fill[1]:.2e}")

splits = ["random_stratified", "family_holdout"]
beh_models = {s: E.load_behavioral(s) for s in splits}

# ── Training features: out-of-fold Behavioural + corpus-mean K/S (main()'s recipe) ──
print("Building OOF training score space …")
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

# ── Test-set score matrices for the splits we need ──────────────────────────
split_data = {}
for split in splits:
    beh_m_s, beh_sc_s = beh_models[split]
    X_beh_s, y_s, fam_s = E.load_beh_split(split, "test")
    Xs_s = beh_sc_s.transform(X_beh_s).astype(np.float32)
    p_b_s, sig_b_s = E.mc_infer_batched(beh_m_s, Xs_s)
    Xk_s = np.stack([know_lookup(f) for f in fam_s]).astype(np.float32)
    p_k_s, sig_k_s = E.mc_infer_batched(know_m, Xk_s)
    n_s = len(p_b_s)
    p_s_s   = np.full(n_s, sem_fill[0], np.float32)
    sig_s_s = np.full(n_s, sem_fill[1], np.float32)
    p_s_pc, p_b_pc, p_k_pc = E.peer_consult(
        p_s_s.copy(), p_b_s.copy(), p_k_s.copy(),
        sig_s_s.copy(), sig_b_s.copy(), sig_k_s.copy())
    X_te_pc = np.stack([p_s_pc, p_b_pc, p_k_pc,
                         sig_s_s, sig_b_s, sig_k_s], axis=1).astype(np.float32)
    sig_max = np.stack([sig_b_s, sig_k_s]).max(0)
    split_data[split] = {"y": y_s, "fam": fam_s, "X_te_pc": X_te_pc, "sig_max": sig_max}

y_te_rs    = split_data["random_stratified"]["y"]
X_te_score = split_data["random_stratified"]["X_te_pc"]

# ── Train the same CTGAN-augmented Decision MLP main() uses for E1 ──────────
print("Training CTGAN-augmented Decision MLP (seed=0, matching main()'s E1 model) …")
auroc, auprc, ece_v, eval_mlp, eval_sc = E.train_mlp_variant(
    X_tr_score, y_tr_score, X_te_score, y_te_rs, aug="ctgan", seed=SEED)
eval_mlp.eval()
print(f"  random_stratified test AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ECE={ece_v:.4f}")

def fused(split):
    d = split_data[split]
    probs = E.fuse_mlp(eval_mlp, eval_sc, 1.0, d["X_te_pc"])
    return probs, d["sig_max"], d["y"], d["fam"]

p_hat_rs, sig_max_rs, y_rs, fam_rs = fused("random_stratified")
p_hat_fh, sig_max_fh, y_fh, fam_fh = fused("family_holdout")

from sklearn.metrics import roc_auc_score
print("  sanity re-check AUROC:", round(roc_auc_score(y_rs, p_hat_rs), 4))

HELD_OUT_FAMILIES = {"REvil", "WannaCry"}
ood_mask = np.isin(fam_fh, list(HELD_OUT_FAMILIES)) & (y_fh == 1)

ransom_mask = y_rs == 1
benign_mask = y_rs == 0

out = {
    "policy": policy,
    "mlp_seed": SEED,
    "random_stratified_test_auroc": round(float(auroc), 4),
    "ransomware_train_dist": {
        "p_hat": p_hat_rs[ransom_mask].tolist(),
        "sigma_max": sig_max_rs[ransom_mask].tolist(),
        "n": int(ransom_mask.sum()),
    },
    "benign_train_dist": {
        "p_hat": p_hat_rs[benign_mask].tolist(),
        "sigma_max": sig_max_rs[benign_mask].tolist(),
        "n": int(benign_mask.sum()),
    },
    "unseen_family_ood": {
        "p_hat": p_hat_fh[ood_mask].tolist(),
        "sigma_max": sig_max_fh[ood_mask].tolist(),
        "n": int(ood_mask.sum()),
        "families": sorted(set(fam_fh[ood_mask].tolist())),
    },
}
print("ransomware n:", out["ransomware_train_dist"]["n"])
print("benign n:", out["benign_train_dist"]["n"])
print("unseen n:", out["unseen_family_ood"]["n"], out["unseen_family_ood"]["families"])

out_path = E.RESULTS / "triage_space_scatter.json"
out_path.write_text(json.dumps(out, indent=2))
print("saved ->", out_path)
