# tune_policy.py
"""
Automatic calibration of the uncertainty-aware triage policy (τ, κ, τ_high, κ_low)
for Agentic SABRE.

Steps:
 1. Load semantic + behavioural datasets.
 2. Use trained SemanticAgent, BehavioralAgent, and DecisionAgent to compute:
      - fused risk p_hat
      - epistemic uncertainty sigma_max
      - ground-truth labels y
 3. Sweep τ and κ over sensible grids.
 4. Apply the SABRE triage policy:
        AUTO_CONTAIN if p_hat >= τ_high and sigma_max <= κ_low
        ESCALATE    if p_hat >= τ or sigma_max > κ
        ALLOW       otherwise
 5. For each (τ, κ), compute:
      - balanced accuracy (TPR+TNR)/2 where "alarm" = {AUTO_CONTAIN, ESCALATE}
      - escalation rate = fraction of samples not ALLOW
 6. Pick the (τ, κ) that maximises balanced accuracy;
    break ties by choosing the lower escalation rate.
 7. Derive τ_high and κ_low via simple stricter constraints:
        τ_high = min(0.99, τ + 0.10)
        κ_low  = 0.5 * κ
 8. Save the resulting policy to checkpoints/policy_auto.json
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
    """
    Build index pairs (i_sem, j_beh) such that labels match as often as possible.
    Returns two lists: sem_indices, beh_indices of equal length.

    This mirrors the idea used in training the Decision MLP: we approximate
    a joint sample by coupling semantic and behavioural instances with the
    same label when possible.
    """
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

    # semantic-driven pairs
    for i in range(len(y_sem)):
        yi = int(y_sem_int[i])
        cand = idx_beh[yi]
        if len(cand) > 0:
            j = int(rng.choice(cand))
        else:
            j = int(rng.integers(0, len(y_beh)))
        sem_indices.append(i)
        beh_indices.append(j)

    # behavioural-driven pairs
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
    """
    Vectorised application of the SABRE triage policy:
      AUTO_CONTAIN if p_hat >= τ_high and sigma_max <= κ_low
      ESCALATE    if p_hat >= τ or sigma_max > κ
      ALLOW       otherwise
    Returns an array of strings: "AUTO_CONTAIN" | "ESCALATE" | "ALLOW".
    """
    p_hat = np.asarray(p_hat)
    sigma_max = np.asarray(sigma_max)

    decisions = np.full(p_hat.shape, "ALLOW", dtype=object)

    auto_mask = (p_hat >= tau_high) & (sigma_max <= kappa_low)
    decisions[auto_mask] = "AUTO_CONTAIN"

    esc_mask = (~auto_mask) & ((p_hat >= tau) | (sigma_max > kappa))
    decisions[esc_mask] = "ESCALATE"

    return decisions


def evaluate_policy(p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low):
    """
    Given p_hat, sigma_max, and labels y, evaluate the triage policy for
    a particular (τ, κ, τ_high, κ_low) configuration.

    We treat "alarm" = {AUTO_CONTAIN, ESCALATE}, "no alarm" = ALLOW.
    Returns:
      - balanced_accuracy
      - escalation_rate
      - confusion matrix (2x2, alarm vs no-alarm)
    """
    y = np.asarray(y).astype(int)
    decisions = sabre_decisions(p_hat, sigma_max, tau, kappa, tau_high, kappa_low)

    # alarm = 1 if decision != ALLOW
    pred_alarm = (decisions != "ALLOW").astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred_alarm, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (tpr + tnr)
    escalation_rate = float(pred_alarm.mean())

    return bal_acc, escalation_rate, (int(tn), int(fp), int(fn), int(tp)), decisions


def main():
    rng = np.random.default_rng(123)

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

    # 2) Instantiate agents
    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    dec_agent = DecisionAgent(
        artifact_path="checkpoints/decision_fuser.json",
        use_uncertainties=True,
    )

    # 3) Build joint index pairs (semantic, behavioural) with matching labels
    sem_idx, beh_idx = build_joint_pairs(y_sem, y_beh, rng)

    # 4) For each pair, compute p_sem, σ_sem, p_beh, σ_beh, fused p_hat, sigma_max, and label
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

        # label: take semantic label for first half, behavioural label for second half
        # (they are matched by class anyway)
        if len(y_list) < len(sem_idx):
            y_list.append(y_sem[i_s])
        else:
            y_list.append(y_beh[i_b])

    p_hat = np.asarray(p_hat_list, dtype=np.float32)
    sigma_max = np.asarray(sigma_max_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # 5) Define search grids for τ and κ based on empirical quantiles
    tau_grid = np.quantile(p_hat, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    kappa_grid = np.quantile(sigma_max, [0.2, 0.4, 0.6, 0.8])

    best = None

    for tau in tau_grid:
        for kappa in kappa_grid:
            # stricter thresholds for AUTO_CONTAIN
            tau_high = float(min(0.99, tau + 0.10))
            kappa_low = float(0.5 * kappa)

            bal_acc, esc_rate, (tn, fp, fn, tp), _ = evaluate_policy(
                p_hat, sigma_max, y, tau, kappa, tau_high, kappa_low
            )

            cand = {
                "tau": float(tau),
                "kappa": float(kappa),
                "tau_high": tau_high,
                "kappa_low": kappa_low,
                "balanced_accuracy": float(bal_acc),
                "escalation_rate": float(esc_rate),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }

            if best is None:
                best = cand
            else:
                # keep the one with higher balanced accuracy; tie-break on lower escalation rate
                if cand["balanced_accuracy"] > best["balanced_accuracy"] + 1e-4:
                    best = cand
                elif abs(cand["balanced_accuracy"] - best["balanced_accuracy"]) <= 1e-4:
                    if cand["escalation_rate"] < best["escalation_rate"]:
                        best = cand

    # 6) Save best policy to JSON
    out_path = Path("checkpoints/policy_auto.json")
    out_path.write_text(json.dumps(best, indent=2))
    print("=== Best policy found ===")
    print(json.dumps(best, indent=2))
    print(f"Saved to {out_path}")
    

if __name__ == "__main__":
    main()
