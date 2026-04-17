# plot_calibration_fusion.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import torch
from Agentic_SABRE.agents.decision import FusionMLP
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

    idx_sem = {0: np.where(y_sem_int == 0)[0],
               1: np.where(y_sem_int == 1)[0]}
    idx_beh = {0: np.where(y_beh_int == 0)[0],
               1: np.where(y_beh_int == 1)[0]}

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


def main():
    rng = np.random.default_rng(42)

    df_sem = pd.read_parquet(
        r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
    )
    df_beh = pd.read_parquet(
        r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
    )

    y_sem = df_sem["label"].astype(np.float32).values
    y_beh = df_beh["label"].astype(np.float32).values

    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    dec_agent = DecisionAgent(
        artifact_path="checkpoints/decision_fuser.json",
        use_uncertainties=True,
    )

    # Load fusion artifact manually to get uncalibrated model
    blob = json.loads(Path("checkpoints/decision_fuser.json").read_text())
    in_dim = int(blob["in_dim"])
    scaler_mean = np.array(blob["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(blob["scaler_scale"], dtype=np.float32)
    cfg = blob["cfg"]
    hidden = cfg.get("hidden", 32)
    p_drop = cfg.get("p_drop", 0.2)

    fusion_raw = FusionMLP(in_dim, hidden=hidden, p_drop=p_drop)
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["model_state"].items()}
    fusion_raw.load_state_dict(state, strict=True)
    fusion_raw.eval()

    # Build joint samples
    sem_idx, beh_idx = build_joint_pairs(y_sem, y_beh, rng)

    X_rows = []
    y_rows = []

    n_sem = len(y_sem)
    for i_s, i_b in zip(sem_idx[:n_sem], beh_idx[:n_sem]):
        out_sem = sem_agent.score({"embedding": X_sem[i_s]})
        out_beh = beh_agent.score({"features": X_beh[i_b]})
        X_rows.append([out_sem.prob, out_beh.prob, out_sem.uncert, out_beh.uncert])
        y_rows.append(y_sem[i_s])

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    # scale
    X_scaled = (X - scaler_mean) / scaler_scale

    with torch.no_grad():
        p_uncal = fusion_raw(torch.from_numpy(X_scaled)).numpy().reshape(-1)

    p_cal = []
    for (p_s, p_b, s_s, s_b) in X_rows:
        fused = dec_agent.score({
            "p_sem": p_s,
            "p_beh": p_b,
            "σ_sem": s_s,
            "σ_beh": s_b,
        })
        p_cal.append(fused.prob)
    p_cal = np.asarray(p_cal, dtype=np.float32)

    # Reliability curves
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y, p_uncal, n_bins=10, strategy="uniform")
    frac_pos_cal, mean_pred_cal = calibration_curve(y, p_cal, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(mean_pred_uncal, frac_pos_uncal, marker="o", label="Uncalibrated")
    plt.plot(mean_pred_cal, frac_pos_cal, marker="s", label="Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram (fusion)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Brier (uncal):", brier_score_loss(y, p_uncal))
    print("Brier (calibrated):", brier_score_loss(y, p_cal))


if __name__ == "__main__":
    main()
