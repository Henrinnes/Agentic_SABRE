# tune_policy_low_escalation.py
"""
Low-escalation policy calibration for Agentic SABRE.

We reuse the same fused risk p_hat, epistemic uncertainty sigma_max, and labels y
as in tune_policy.py, but we choose (τ, κ, τ_high, κ_low) to optimise:

    objective = balanced_accuracy - λ * escalation_rate

where:
  - balanced_accuracy is computed with "alarm" = {AUTO_CONTAIN, ESCALATE},
  - escalation_rate is the fraction of samples that are not ALLOW,
  - λ controls how strongly we penalise escalation (here λ = 0.5).

The result is a more automation-friendly policy that trades a bit of safety margin
for reduced analyst workload.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.agents.decision import DecisionAgent

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def build_joint_pairs(y_sem, y_beh, rng):
    y_sem_int = y_sem.astype(int)
    y_beh_int = y_beh.astype(int)

    idx_sem = {
        0: np.where(y_sem_int == 0)[0],
        1: np.where(y_sem_int == 1)[0],
    }
    idx_beh = {
        0: np.where(y_beh_int == 0)[0],
        1: np.where(y_beh_int == 1)[0],
    }

    sem_indices = []
    beh_indices = []

    for i in range(len(y_sem)):
        yi = int(y_sem_int[i])
        cand = idx_beh[yi]
        if len(cand) > 0:
            j = int(rng.choice(cand))
        else:
            j = int(rng.integers(0, len(y_beh)))
        sem_indices.append(i)
        beh_indices.append(j)

    for j in range(len(y_beh)):
        yj = int(y_beh_int[j])
        cand = idx_sem[yj]
        if len(cand) > 0:
            i = int(rng.choice(cand))
        else:
            i = int(rng.integers(0, len(y_sem)))
        sem_indices.append(i)
        beh_indices.append(j)

    return sem_indices, beh_indices


def sabre_decisions(p_hat, sigma_max, tau, kappa, tau_high, kappa_low):
    p_hat = np.asarray(p_hat)
    sigma_max = np.asarray(sigma_max)

    decisions = np.full(p_hat.shape, "ALLOW", dtype=object)

    auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
    decisions[auto_mask] = "AUTO_CONTAIN"

    esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
    decisions[esc_mask] = "ESCALATE"

    return decisions


def evaluate_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low):
    y = np.asarray(y).astype(int)
    decisions = sabre_decisions(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)

    pred_alarm = (decisions != "ALLOW").astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred_alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (tpr + tnr)
    escalation_rate = float(pred_alarm.mean())

    return bal_acc, escalation_rate, (int(tn), int(fp), int(fn), int(tp)), decisions


def main():
    rng = np.random.default_rng(456)

    # 1) Load datasets
    df_sem = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet"
    )
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet"
    )

    y_sem = df_sem["label"].astype(np.float32).values
    y_beh = df_beh["label"].astype(np.float32).values

    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

    # 2) Agents
    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    dec_agent = DecisionAgent(
        artifact_path="checkpoints/decision_fuser.json",
        use_uncertainties=True,
    )

    # 3) Joint pairs for calibration
    sem_idx, beh_idx = build_joint_pairs(y_sem, y_beh, rng)

    p_hat_list = []
    sigma_max_list = []
    y_list = []

    for i_s, i_b in zip(sem_idx, beh_idx):
        out_sem = sem_agent.score({"embedding": X_sem[i_s]})
        out_beh = beh_agent.score({"features": X_beh[i_b]})

        fused = dec_agent.score({
            "p_sem": out_sem.prob,
            "p_beh": out_beh.prob,
            "σ_sem": out_sem.uncert,
            "σ_beh": out_beh.uncert,
        })

        p_hat_list.append(fused.prob)
        sigma_max_list.append(max(out_sem.uncert, out_beh.uncert))

        if len(y_list) < len(sem_idx):
            y_list.append(y_sem[i_s])
        else:
            y_list.append(y_beh[i_b])

    p_hat = np.asarray(p_hat_list, dtype=np.float32)
    sigma_max = np.asarray(sigma_max_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # 4) Grids over τ and κ based on quantiles (same idea as before)
    tau_grid = np.quantile(p_hat, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    kappa_grid = np.quantile(sigma_max, [0.1, 0.3, 0.5, 0.7])

    best = None
    lambda_penalty = 0.5  # weight for escalation penalty

    for tau in tau_grid:
        for kappa in kappa_grid:
            tau_high = float(min(0.99, tau + 0.10))
            kappa_low = float(0.5 * kappa)

            bal_acc, esc_rate, (tn, fp, fn, tp), _ = evaluate_policy(
                p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low
            )

            objective = bal_acc - lambda_penalty * esc_rate

            cand = {
                "tau": float(tau),
                "kappa": float(kappa),
                "tau_high": tau_high,
                "kappa_low": kappa_low,
                "balanced_accuracy": float(bal_acc),
                "escalation_rate": float(esc_rate),
                "objective": float(objective),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }

            if best is None or cand["objective"] > best["objective"] + 1e-6:
                best = cand

    out_path = Path("checkpoints/policy_low_escalation.json")
    out_path.write_text(json.dumps(best, indent=2))
    print("=== Low-escalation policy found ===")
    print(json.dumps(best, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
