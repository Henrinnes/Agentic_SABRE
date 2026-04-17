# plot_roc_curves.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import roc_curve, auc

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


def main():
    rng = np.random.default_rng(123)

    # --- load data ---
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

    # --- instantiate agents ---
    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    dec_agent = DecisionAgent(
        artifact_path="checkpoints/decision_fuser.json",
        use_uncertainties=True,
    )

    # --- base agent scores ---
    print("Computing base agent scores...")
    p_sem = []
    for z in X_sem:
        out = sem_agent.score({"embedding": z})
        p_sem.append(out.prob)
    p_sem = np.asarray(p_sem, dtype=np.float32)

    p_beh = []
    for b in X_beh:
        out = beh_agent.score({"features": b})
        p_beh.append(out.prob)
    p_beh = np.asarray(p_beh, dtype=np.float32)

    # --- fused scores using joint pairs ---
    print("Computing fused scores...")
    sem_idx, beh_idx = build_joint_pairs(y_sem, y_beh, rng)

    fused_probs = []
    fused_labels = []

    # first half: semantic-driven (RDset-like)
    n_sem = len(y_sem)
    for i_s, i_b in zip(sem_idx[:n_sem], beh_idx[:n_sem]):
        out_sem = sem_agent.score({"embedding": X_sem[i_s]})
        out_beh = beh_agent.score({"features": X_beh[i_b]})
        fused = dec_agent.score({
            "p_sem": out_sem.prob,
            "p_beh": out_beh.prob,
            "σ_sem": out_sem.uncert,
            "σ_beh": out_beh.uncert,
        })
        fused_probs.append(fused.prob)
        fused_labels.append(y_sem[i_s])
    p_fused_rd = np.asarray(fused_probs, dtype=np.float32)
    y_rd = np.asarray(fused_labels, dtype=np.float32)

    # second half: behavioural-driven (RanSMAP-like)
    fused_probs = []
    fused_labels = []
    for i_s, i_b in zip(sem_idx[n_sem:], beh_idx[n_sem:]):
        out_sem = sem_agent.score({"embedding": X_sem[i_s]})
        out_beh = beh_agent.score({"features": X_beh[i_b]})
        fused = dec_agent.score({
            "p_sem": out_sem.prob,
            "p_beh": out_beh.prob,
            "σ_sem": out_sem.uncert,
            "σ_beh": out_beh.uncert,
        })
        fused_probs.append(fused.prob)
        fused_labels.append(y_beh[i_b])
    p_fused_beh = np.asarray(fused_probs, dtype=np.float32)
    y_ransmap = np.asarray(fused_labels, dtype=np.float32)

    # --- ROC curves ---
    print("Plotting ROC curves...")

    # RDset: semantic vs fusion
    fpr_s, tpr_s, _ = roc_curve(y_sem, p_sem)
    auc_s = auc(fpr_s, tpr_s)
    fpr_fr, tpr_fr, _ = roc_curve(y_rd, p_fused_rd)
    auc_fr = auc(fpr_fr, tpr_fr)

    plt.figure()
    plt.plot(fpr_s, tpr_s, label=f"Semantic (AUC={auc_s:.3f})")
    plt.plot(fpr_fr, tpr_fr, label=f"Fusion (AUC={auc_fr:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC on RDset")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # RanSMAP: behavioural vs fusion
    fpr_b, tpr_b, _ = roc_curve(y_beh, p_beh)
    auc_b = auc(fpr_b, tpr_b)
    fpr_fb, tpr_fb, _ = roc_curve(y_ransmap, p_fused_beh)
    auc_fb = auc(fpr_fb, tpr_fb)

    plt.figure()
    plt.plot(fpr_b, tpr_b, label=f"Behavioural (AUC={auc_b:.3f})")
    plt.plot(fpr_fb, tpr_fb, label=f"Fusion (AUC={auc_fb:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC on RanSMAP")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
