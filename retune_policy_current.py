#!/usr/bin/env python3
"""Retune the triage policy (tau, kappa, tau_high, kappa_low) against a
Decision MLP trained fresh via eval_sabre2.py's own recipe (out-of-fold
Behavioural + corpus-mean Semantic/Knowledge + CTGAN, matching main()'s E1
model), instead of the stale on-disk checkpoints/decision_fuser_3agent.json
(see gen_triage_space_data.py's docstring for why that artifact is stale).

Persists the full Optuna trial history via --storage, unlike the original
tune_policy_optuna.py run that produced checkpoints/policy_auto.json.

Usage:
    python retune_policy_current.py --out checkpoints/policy_auto_v2.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import confusion_matrix

os.chdir(Path(__file__).parent)
import eval_sabre2 as E

optuna.logging.set_verbosity(optuna.logging.WARNING)


def apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low):
    p_hat = np.asarray(p_hat)
    sigma_max = np.asarray(sigma_max)
    decisions = np.full(len(p_hat), "ALLOW", dtype=object)
    auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
    decisions[auto_mask] = "AUTO_CONTAIN"
    esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
    decisions[esc_mask] = "ESCALATE"
    return decisions


def score_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low):
    decisions = apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)
    alarm = (decisions != "ALLOW").astype(int)
    y = np.asarray(y, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    return 0.5 * (tpr + tnr)


def main():
    parser = argparse.ArgumentParser(description="Retune triage policy against a current Decision MLP")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0,
                         help="Decision MLP training seed (default 0, matching gen_triage_space_data.py)")
    parser.add_argument("--out", type=str, default="checkpoints/policy_auto_v2.json")
    parser.add_argument("--storage", type=str, default="sqlite:///checkpoints/optuna_studies.db")
    parser.add_argument("--study-name", type=str, default=None)
    args = parser.parse_args()

    print("Loading current models (semantic, knowledge) ...")
    sem_m = E.load_semantic()
    know_m, know_sc = E.load_knowledge()
    know_lookup = E.build_knowledge_lookup(know_m, know_sc)
    _, mlp_sc_ref, _, _ = E.load_decision_mlp()
    sem_fill = E.build_semantic_fill(sem_m)
    sem_fill = (sem_fill[0], float(mlp_sc_ref.mean_[3]))
    print(f"  Semantic fill: p_s={sem_fill[0]:.4f}  sig_s={sem_fill[1]:.2e}")

    print("Loading behavioural models (random_stratified) ...")
    beh_m_rs, beh_sc_rs = E.load_behavioral("random_stratified")

    print("Building out-of-fold training score space ...")
    p_b_tr, sig_b_tr, y_tr, fam_tr = E.build_oof_behavioural_scores()
    Xk_tr = np.stack([know_lookup(f) for f in fam_tr]).astype(np.float32)
    p_k_tr, sig_k_tr = E.mc_infer_batched(know_m, Xk_tr)
    n_tr = len(p_b_tr)
    p_s_tr = np.full(n_tr, sem_fill[0], np.float32)
    sig_s_tr = np.full(n_tr, sem_fill[1], np.float32)
    p_s_tr_pc, p_b_tr_pc, p_k_tr_pc = E.peer_consult(
        p_s_tr, p_b_tr, p_k_tr, sig_s_tr, sig_b_tr, sig_k_tr)
    X_tr_score = np.stack(
        [p_s_tr_pc, p_b_tr_pc, p_k_tr_pc, sig_s_tr, sig_b_tr, sig_k_tr], axis=1
    ).astype(np.float32)
    y_tr_score = y_tr.astype(np.float32)

    print("Building random_stratified test score space ...")
    X_beh_te, y_te, fam_te = E.load_beh_split("random_stratified", "test")
    Xs_te = beh_sc_rs.transform(X_beh_te).astype(np.float32)
    p_b_te, sig_b_te = E.mc_infer_batched(beh_m_rs, Xs_te)
    Xk_te = np.stack([know_lookup(f) for f in fam_te]).astype(np.float32)
    p_k_te, sig_k_te = E.mc_infer_batched(know_m, Xk_te)
    n_te = len(p_b_te)
    p_s_te = np.full(n_te, sem_fill[0], np.float32)
    sig_s_te = np.full(n_te, sem_fill[1], np.float32)
    p_s_te_pc, p_b_te_pc, p_k_te_pc = E.peer_consult(
        p_s_te.copy(), p_b_te.copy(), p_k_te.copy(),
        sig_s_te.copy(), sig_b_te.copy(), sig_k_te.copy())
    X_te_score = np.stack(
        [p_s_te_pc, p_b_te_pc, p_k_te_pc, sig_s_te, sig_b_te, sig_k_te], axis=1
    ).astype(np.float32)
    sig_max_te = np.stack([sig_b_te, sig_k_te]).max(0)

    print(f"Training CTGAN-augmented Decision MLP (seed={args.seed}, matching main()'s E1 model) ...")
    auroc, auprc, ece_v, eval_mlp, eval_sc = E.train_mlp_variant(
        X_tr_score, y_tr_score, X_te_score, y_te, aug="ctgan", seed=args.seed)
    eval_mlp.eval()
    print(f"  random_stratified test AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ECE={ece_v:.4f}")

    p_hat = E.fuse_mlp(eval_mlp, eval_sc, 1.0, X_te_score)
    sigma_max = sig_max_te
    y = y_te

    print(f"  p_hat  : min={p_hat.min():.3f}  max={p_hat.max():.3f}  "
          f"mean={p_hat.mean():.3f}  median={np.median(p_hat):.3f}")
    print(f"  sigma  : min={sigma_max.min():.4f}  max={sigma_max.max():.4f}  "
          f"mean={sigma_max.mean():.4f}")

    p_low, p_high = float(np.quantile(p_hat, 0.10)), float(np.quantile(p_hat, 0.95))
    s_low, s_high = float(np.quantile(sigma_max, 0.05)), float(np.quantile(sigma_max, 0.90))

    def objective(trial):
        tau = trial.suggest_float("tau", p_low, p_high)
        kappa = trial.suggest_float("kappa", s_low, s_high)
        tau_high = trial.suggest_float("tau_high", tau, min(0.999, p_high + 0.05))
        kappa_low = trial.suggest_float("kappa_low", max(1e-4, s_low * 0.5), kappa)
        return score_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low)

    study_name = args.study_name or Path(args.out).stem
    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning Optuna ({args.trials} trials, study '{study_name}' -> {args.storage}) ...")
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    tau, kappa, tau_high, kappa_low = (
        best["tau"], best["kappa"], best["tau_high"], best["kappa_low"]
    )

    decisions = apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)
    alarm = (decisions != "ALLOW").astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (tpr + tnr)
    esc_rate = float(alarm.mean())

    result = {
        "tau": round(tau, 6),
        "kappa": round(kappa, 6),
        "tau_high": round(tau_high, 6),
        "kappa_low": round(kappa_low, 6),
        "balanced_accuracy": round(float(bal_acc), 6),
        "escalation_rate": round(float(esc_rate), 6),
        "mlp_seed": args.seed,
        "mlp_auroc_random_stratified": round(float(auroc), 4),
        "n_trials": args.trials,
        "n_samples": int(n_te),
        "triage": {
            "AUTO_CONTAIN": int((decisions == "AUTO_CONTAIN").sum()),
            "ESCALATE": int((decisions == "ESCALATE").sum()),
            "ALLOW": int((decisions == "ALLOW").sum()),
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
