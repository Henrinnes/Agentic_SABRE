# tune_policy_optuna.py
"""
Bayesian optimisation of the SABRE triage policy thresholds
(τ, κ, τ_high, κ_low) using Optuna (TPE sampler).

Supports both 2-agent and 3-agent pipelines.
Uses GPU-batched inference for speed.

Objective: maximise balanced accuracy over the triage alarm decisions,
           with an optional penalty on escalation rate.

Usage
-----
    python tune_policy_optuna.py                        # 3-agent, 300 trials, saves policy_auto.json
    python tune_policy_optuna.py --pipeline 2           # 2-agent
    python tune_policy_optuna.py --trials 500           # more trials
    python tune_policy_optuna.py --out policy_optuna.json
    python tune_policy_optuna.py --esc-penalty 0.3      # penalise escalation rate

Output
------
    checkpoints/policy_auto.json  (or --out path)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import confusion_matrix

optuna.logging.set_verbosity(optuna.logging.WARNING)

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOW_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal", "Family_Count", "Ransomware_Context",
    "label",
]

BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def stratified_sample(df, n_per_class, seed=42):
    parts = [
        df[df.label == lbl].sample(
            min(n_per_class, (df.label == lbl).sum()), random_state=seed
        )
        for lbl in [0, 1]
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def stratified_sample_by(df, label_col, n_per_class, seed=42):
    parts = [
        df[df[label_col] == lbl].sample(
            min(n_per_class, (df[label_col] == lbl).sum()), random_state=seed
        )
        for lbl in [0, 1]
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Policy evaluation (vectorised)
# ---------------------------------------------------------------------------

def apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low):
    p_hat = np.asarray(p_hat)
    sigma_max = np.asarray(sigma_max)
    decisions = np.full(len(p_hat), "ALLOW", dtype=object)
    auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
    decisions[auto_mask] = "AUTO_CONTAIN"
    esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
    decisions[esc_mask] = "ESCALATE"
    return decisions


def score_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low, esc_penalty):
    decisions = apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)
    alarm = (decisions != "ALLOW").astype(int)
    y = np.asarray(y, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (tpr + tnr)
    esc_rate = float(alarm.mean())
    return bal_acc - esc_penalty * esc_rate


# ---------------------------------------------------------------------------
# Score collection (batched GPU inference)
# ---------------------------------------------------------------------------

def collect_scores_2agent(orch, X_sem, y_sem, X_beh, y_beh, rng):
    """Pair semantic and behavioural samples by label, run batched inference."""
    # Match labels
    idx_beh_by_label = {
        lbl: np.where(y_beh.astype(int) == lbl)[0]
        for lbl in [0, 1]
    }
    beh_idx = []
    for yi in y_sem.astype(int):
        cands = idx_beh_by_label[yi]
        beh_idx.append(int(rng.choice(cands)) if len(cands) > 0
                       else int(rng.integers(0, len(y_beh))))
    beh_idx = np.array(beh_idx)

    N = len(y_sem)
    inputs_sem  = [{"embedding": X_sem[i]}       for i in range(N)]
    inputs_beh  = [{"features":  X_beh[beh_idx[i]]} for i in range(N)]

    p_hats, sigma_maxs = [], []
    for start in range(0, N, BATCH_SIZE):
        batch = orch.handle_batch(
            inputs_sem[start:start + BATCH_SIZE],
            inputs_beh[start:start + BATCH_SIZE],
        )
        for r in batch:
            p_hats.append(r["score_fused"])
            sigma_maxs.append(r["uncert_max"])

    return np.array(p_hats, dtype=np.float32), np.array(sigma_maxs, dtype=np.float32), y_sem


def collect_scores_3agent(orch, X_sem, y_sem, X_beh, y_beh, X_know, y_know, rng):
    """Match labels across all three datasets, run batched inference."""
    idx_beh_by_label  = {lbl: np.where(y_beh.astype(int)  == lbl)[0] for lbl in [0, 1]}
    idx_know_by_label = {lbl: np.where(y_know.astype(int) == lbl)[0] for lbl in [0, 1]}

    beh_idx, know_idx = [], []
    for yi in y_sem.astype(int):
        cb = idx_beh_by_label[yi]
        ck = idx_know_by_label[yi]
        beh_idx.append(int(rng.choice(cb))  if len(cb) > 0 else int(rng.integers(0, len(y_beh))))
        know_idx.append(int(rng.choice(ck)) if len(ck) > 0 else int(rng.integers(0, len(y_know))))
    beh_idx  = np.array(beh_idx)
    know_idx = np.array(know_idx)

    N = len(y_sem)
    inputs_sem  = [{"embedding": X_sem[i]}             for i in range(N)]
    inputs_beh  = [{"features":  X_beh[beh_idx[i]]}   for i in range(N)]
    inputs_know = [{"features":  X_know[know_idx[i]]}  for i in range(N)]

    p_hats, sigma_maxs = [], []
    for start in range(0, N, BATCH_SIZE):
        batch = orch.handle_batch(
            inputs_sem[start:start + BATCH_SIZE],
            inputs_beh[start:start + BATCH_SIZE],
            inputs_know[start:start + BATCH_SIZE],
        )
        for r in batch:
            p_hats.append(r["score_fused"])
            sigma_maxs.append(r["uncert_max"])

    return np.array(p_hats, dtype=np.float32), np.array(sigma_maxs, dtype=np.float32), y_sem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna policy threshold optimisation")
    parser.add_argument("--pipeline",    type=int,   default=3,    choices=[2, 3],
                        help="2-agent or 3-agent pipeline (default: 3)")
    parser.add_argument("--trials",      type=int,   default=300,
                        help="Number of Optuna trials (default: 300)")
    parser.add_argument("--esc-penalty", type=float, default=0.0,
                        help="Escalation-rate penalty weight (default: 0 = balance-acc only)")
    parser.add_argument("--out",         type=str,   default="checkpoints/policy_auto.json",
                        help="Output JSON path")
    parser.add_argument("--n-samples",   type=int,   default=2000,
                        help="Semantic samples per class for tuning (default: 2000)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    #  Load datasets
    # ------------------------------------------------------------------ #
    print("Loading datasets...")
    df_sem = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet")
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet")

    sem_s = stratified_sample(df_sem, args.n_samples)
    beh_s = stratified_sample(df_beh, args.n_samples)

    X_sem = np.stack(sem_s["embedding"].values).astype(np.float32)
    y_sem = sem_s["label"].astype(int).values
    X_beh = beh_s[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y_beh = beh_s["label"].astype(int).values

    if args.pipeline == 3:
        df_know_raw = pd.read_csv(
            r"C:\Users\henri\SampleDatasets\Dataset_RansKnow\outputs\Knowledge_Agent_Features_307.csv")
        df_know_raw["label"] = (
            (df_know_raw["Tactic_Total_Mentions"] > 10) |
            (df_know_raw["Tool_Total_Mentions"] > 2)
        ).astype(int)
        know_s = stratified_sample_by(df_know_raw, "label", 100)
        X_know = (
            know_s.drop(columns=KNOW_DROP_COLS, errors="ignore")
                  .select_dtypes(include="number")
                  .to_numpy(dtype=np.float32)
        )
        y_know = know_s["label"].astype(int).values

    # ------------------------------------------------------------------ #
    #  Load orchestrator
    # ------------------------------------------------------------------ #
    print(f"Loading {args.pipeline}-agent pipeline...")
    if args.pipeline == 2:
        from Agentic_SABRE.orchestrator import Orchestrator
        orch = Orchestrator(decision_artifact="checkpoints/decision_fuser.json")
    else:
        from Agentic_SABRE.orchestrator_3agent import Orchestrator
        orch = Orchestrator(decision_artifact="checkpoints/decision_fuser_3agent.json")

    # ------------------------------------------------------------------ #
    #  Run GPU-batched inference once to collect (p_hat, sigma_max, y)
    # ------------------------------------------------------------------ #
    print("Running batched inference to collect risk scores...")
    t0 = time.time()
    if args.pipeline == 2:
        p_hat, sigma_max, y = collect_scores_2agent(orch, X_sem, y_sem, X_beh, y_beh, rng)
    else:
        p_hat, sigma_max, y = collect_scores_3agent(orch, X_sem, y_sem, X_beh, y_beh, X_know, y_know, rng)
    elapsed = time.time() - t0

    N = len(y)
    print(f"  Collected {N} samples in {elapsed:.1f}s  "
          f"({N/elapsed:.0f} samples/s)")
    print(f"  p_hat  : min={p_hat.min():.3f}  max={p_hat.max():.3f}  "
          f"mean={p_hat.mean():.3f}  median={np.median(p_hat):.3f}")
    print(f"  sigma  : min={sigma_max.min():.4f}  max={sigma_max.max():.4f}  "
          f"mean={sigma_max.mean():.4f}")

    # Search bounds derived from empirical distribution
    p_low,  p_high  = float(np.quantile(p_hat,   0.10)), float(np.quantile(p_hat,   0.95))
    s_low,  s_high  = float(np.quantile(sigma_max, 0.05)), float(np.quantile(sigma_max, 0.90))

    # ------------------------------------------------------------------ #
    #  Optuna objective
    # ------------------------------------------------------------------ #
    def objective(trial):
        tau       = trial.suggest_float("tau",       p_low,  p_high)
        kappa     = trial.suggest_float("kappa",     s_low,  s_high)
        tau_high  = trial.suggest_float("tau_high",  tau,    min(0.999, p_high + 0.05))
        kappa_low = trial.suggest_float("kappa_low", max(1e-4, s_low * 0.5), kappa)

        return score_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low,
                            args.esc_penalty)

    # ------------------------------------------------------------------ #
    #  Run optimisation
    # ------------------------------------------------------------------ #
    print(f"\nRunning Optuna ({args.trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    tau       = best["tau"]
    kappa     = best["kappa"]
    tau_high  = best["tau_high"]
    kappa_low = best["kappa_low"]

    # Final evaluation with winning thresholds
    decisions = apply_policy(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)
    alarm = (decisions != "ALLOW").astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc    = 0.5 * (tpr + tnr)
    esc_rate   = float(alarm.mean())
    auto_count = int((decisions == "AUTO_CONTAIN").sum())
    esc_count  = int((decisions == "ESCALATE").sum())
    allow_count= int((decisions == "ALLOW").sum())

    result = {
        "tau":               round(tau,       6),
        "kappa":             round(kappa,     6),
        "tau_high":          round(tau_high,  6),
        "kappa_low":         round(kappa_low, 6),
        "balanced_accuracy": round(float(bal_acc),  6),
        "escalation_rate":   round(float(esc_rate), 6),
        "esc_penalty_used":  args.esc_penalty,
        "pipeline":          args.pipeline,
        "n_trials":          args.trials,
        "n_samples":         N,
        "triage": {
            "AUTO_CONTAIN": auto_count,
            "ESCALATE":     esc_count,
            "ALLOW":        allow_count,
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "="*60)
    print("  BEST POLICY (Optuna)")
    print("="*60)
    print(f"  tau       = {tau:.4f}   (escalate if p_hat >= tau)")
    print(f"  kappa     = {kappa:.4f}   (escalate if sigma > kappa)")
    print(f"  tau_high  = {tau_high:.4f}   (auto-contain if p_hat >= tau_high)")
    print(f"  kappa_low = {kappa_low:.4f}   (auto-contain if sigma <= kappa_low)")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print(f"  Escalation rate   : {esc_rate:.4f}")
    print(f"  Triage breakdown  : AUTO_CONTAIN={auto_count}  ESCALATE={esc_count}  ALLOW={allow_count}")
    print(f"  Saved to          : {out_path}")

    return result


if __name__ == "__main__":
    main()
