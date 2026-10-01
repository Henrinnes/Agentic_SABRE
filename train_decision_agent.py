# # train_decision_agent.py
# """
# Train the Decision MLP (fusion head) on score-space:
#  - Inputs: [p_sem, p_beh, σ_sem, σ_beh]
#  - Labels: y ∈ {0,1}
#  - Uses MC Dropout agents to generate scores
#  - Uses CTGAN (via SDV) to augment TRAIN-ONLY score tuples
# This matches the Agentic SABRE diagram: score pool → CTGAN → Decision MLP.
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path

# from Agentic_SABRE.agents.semantic import SemanticAgent
# from Agentic_SABRE.agents.behavioural import BehavioralAgent
# from Agentic_SABRE.agents.decision import DecisionTrainer, DecisionTrainConfig


# # Columns for behavioural features (must match training)
# BEHAVIOR_COLUMNS = [
#     "avg_read_throughput", "avg_write_throughput",
#     "read_LBA_variance", "write_LBA_variance",
#     "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
#     "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
#     "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
#     "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
#     "mem_readwrite_count", "mem_exec_count",
# ]


# def build_joint_score_dataset(
#     p_sem: np.ndarray,
#     s_sem: np.ndarray,
#     y_sem: np.ndarray,
#     p_beh: np.ndarray,
#     s_beh: np.ndarray,
#     y_beh: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Build a realistic joint score dataset X, y where each row looks like:
#       [p_sem, p_beh, σ_sem, σ_beh]

#     We approximate the joint distribution by pairing semantic and behavioural
#     scores with the SAME LABEL when possible:

#       - For each semantic sample i:
#           pick a random behavioural sample k with y_beh[k] == y_sem[i]
#           X_i = [p_sem[i], p_beh[k], s_sem[i], s_beh[k]]

#       - For each behavioural sample j:
#           pick a random semantic sample k with y_sem[k] == y_beh[j]
#           X_j = [p_sem[k], p_beh[j], s_sem[k], s_beh[j]]

#     This way the Decision MLP sees the same 4-D pattern that it will see
#     at inference time, without using artificial zeros.
#     """
#     rng = np.random.default_rng(42)

#     y_sem_int = y_sem.astype(int)
#     y_beh_int = y_beh.astype(int)

#     idx_sem = {
#         0: np.where(y_sem_int == 0)[0],
#         1: np.where(y_sem_int == 1)[0],
#     }
#     idx_beh = {
#         0: np.where(y_beh_int == 0)[0],
#         1: np.where(y_beh_int == 1)[0],
#     }

#     def sample_from(indices: np.ndarray, max_len: int) -> int:
#         if len(indices) > 0:
#             return int(rng.choice(indices))
#         # fallback: sample uniformly from all indices if class is missing
#         return int(rng.integers(0, max_len))

#     X_rows = []
#     y_rows = []

#     # Semantic-driven rows
#     n_sem = len(p_sem)
#     n_beh = len(p_beh)
#     for i in range(n_sem):
#         yi = int(y_sem_int[i])
#         k = sample_from(idx_beh[yi], n_beh)
#         X_rows.append([p_sem[i], p_beh[k], s_sem[i], s_beh[k]])
#         y_rows.append(y_sem[i])

#     # Behaviour-driven rows
#     for j in range(n_beh):
#         yj = int(y_beh_int[j])
#         k = sample_from(idx_sem[yj], n_sem)
#         X_rows.append([p_sem[k], p_beh[j], s_sem[k], s_beh[j]])
#         y_rows.append(y_beh[j])

#     X = np.asarray(X_rows, dtype=np.float32)
#     y = np.asarray(y_rows, dtype=np.float32)
#     return X, y


# def main():
#     # 1) Load datasets
#     df_sem = pd.read_parquet(
#         r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
#     )
#     df_beh = pd.read_parquet(
#         r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
#     )

#     y_sem = df_sem["label"].astype(np.float32).values
#     y_beh = df_beh["label"].astype(np.float32).values

#     X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
#     X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

#     # 2) Instantiate agents with a smaller MC-Dropout T for speed
#     sem_agent = SemanticAgent(T=10)
#     beh_agent = BehavioralAgent(T=10)

#     # 3) Collect semantic scores and uncertainties
#     p_sem, s_sem = [], []
#     for v in X_sem:
#         out = sem_agent.score({"embedding": v})
#         p_sem.append(out.prob)
#         s_sem.append(out.uncert)
#     p_sem = np.asarray(p_sem, dtype=np.float32)
#     s_sem = np.asarray(s_sem, dtype=np.float32)

#     # 4) Collect behavioural scores and uncertainties
#     p_beh, s_beh = [], []
#     for v in X_beh:
#         out = beh_agent.score({"features": v})
#         p_beh.append(out.prob)
#         s_beh.append(out.uncert)
#     p_beh = np.asarray(p_beh, dtype=np.float32)
#     s_beh = np.asarray(s_beh, dtype=np.float32)

#     # 5) Build joint 4-D score dataset [p_sem, p_beh, σ_sem, σ_beh]
#     X, y = build_joint_score_dataset(p_sem, s_sem, y_sem, p_beh, s_beh, y_beh)

#     # 6) Configure DecisionTrainer (uses CTGAN + uncertainties)
#     cfg = DecisionTrainConfig(
#         use_uncertainties=True,   # in_dim = 4 (p_sem, p_beh, σ_sem, σ_beh)
#         ctgan_samples=2000,       # number of synthetic score rows
#         ctgan_epochs=50,          # CTGAN training epochs
#         hidden=32,
#         p_drop=0.2,
#         lr=1e-3,
#         batch_size=128,
#         max_epochs=50,
#         early_stop_patience=6,
#         valid_size=0.2,
#         temperature_scale=True,   # post-hoc calibration
#     )

#     trainer = DecisionTrainer(cfg)
#     trainer.fit(X, y)

#     ckpt_path = Path("checkpoints/decision_fuser.json")
#     trainer.save(str(ckpt_path))
#     print(f"Saved decision fusion artifact to {ckpt_path}")


# if __name__ == "__main__":
#     main()

"""
Train the Decision MLP (fusion head) on score-space.

Inputs:
    [p_sem, p_beh, p_know, σ_sem, σ_beh, σ_know]

Labels:
    y ∈ {0,1}

Uses:
    • MC Dropout agents to generate scores
    • CTGAN (via SDV) to augment TRAIN-ONLY score tuples

Pipeline:
    score pool → CTGAN → Decision MLP
"""
# import inspect
# import Agentic_SABRE.agents.decision as decision_mod

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Dataset locations; override the root with SABRE_DATA_DIR. The RDset parquet is
# built by data_pipeline/rdset_semantic.py (all-MiniLM-L6-v2, Name/md5 dropped first).
_DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
SEM_PARQUET = str(_DATA_DIR / "RDset" / "semantic_dataset_with_embeddings.parquet")
BEH_PARQUET_F = os.environ.get("SABRE_BEH_PARQUET", str(_DATA_DIR / "ransmap_extracted" / "combined_i3-gen12_datasetF.parquet"))
KNOW_CSV_307 = str(_DATA_DIR / "RansKnow" / "Knowledge_Agent_Features_307.csv")

from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.agents.knowledge import KnowledgeAgent
from Agentic_SABRE.agents.decision import DecisionTrainer, DecisionTrainConfig

# print("decision.py loaded from:", decision_mod.__file__)
# print("DecisionTrainer source:")
# print(inspect.getsource(decision_mod.DecisionTrainer.__init__))

# Behaviour feature columns
BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def build_joint_score_dataset(
    p_sem, s_sem, y_sem,
    p_beh, s_beh, y_beh,
    p_know, s_know, y_know,
):
    """
    Build realistic fusion dataset.

    Each row:

    [p_sem, p_beh, p_know, σ_sem, σ_beh, σ_know]
    """

    rng = np.random.default_rng(42)

    y_sem = y_sem.astype(int)
    y_beh = y_beh.astype(int)
    y_know = y_know.astype(int)

    idx_sem = {0: np.where(y_sem == 0)[0], 1: np.where(y_sem == 1)[0]}
    idx_beh = {0: np.where(y_beh == 0)[0], 1: np.where(y_beh == 1)[0]}
    idx_know = {0: np.where(y_know == 0)[0], 1: np.where(y_know == 1)[0]}

    def sample(indices, max_len):
        if len(indices) > 0:
            return int(rng.choice(indices))
        return int(rng.integers(0, max_len))

    X_rows = []
    y_rows = []

    n_sem = len(p_sem)
    n_beh = len(p_beh)
    n_know = len(p_know)

    # semantic-driven rows
    for i in range(n_sem):
        yi = y_sem[i]

        j = sample(idx_beh[yi], n_beh)
        k = sample(idx_know[yi], n_know)

        X_rows.append([
            p_sem[i], p_beh[j], p_know[k],
            s_sem[i], s_beh[j], s_know[k]
        ])

        y_rows.append(yi)

    # behavioural-driven rows
    for j in range(n_beh):
        yj = y_beh[j]

        i = sample(idx_sem[yj], n_sem)
        k = sample(idx_know[yj], n_know)

        X_rows.append([
            p_sem[i], p_beh[j], p_know[k],
            s_sem[i], s_beh[j], s_know[k]
        ])

        y_rows.append(yj)

    # knowledge-driven rows
    for k in range(n_know):
        yk = y_know[k]

        i = sample(idx_sem[yk], n_sem)
        j = sample(idx_beh[yk], n_beh)

        X_rows.append([
            p_sem[i], p_beh[j], p_know[k],
            s_sem[i], s_beh[j], s_know[k]
        ])

        y_rows.append(yk)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    return X, y


def main():

    # -----------------------------
    # Load datasets
    # -----------------------------

    df_sem = pd.read_parquet(
        SEM_PARQUET
    )

    df_beh = pd.read_parquet(
        BEH_PARQUET_F
    )

    df_know = pd.read_csv(
        KNOW_CSV_307
    )

    # labels
    y_sem = df_sem["label"].astype(np.float32).values
    y_beh = df_beh["label"].astype(np.float32).values
    y_know = (df_know["Family_Count"] > 0).astype(np.float32).values

    # inputs
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)

    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

    drop_cols = [
        "Video_ID","Channel_ID","Channel_Name","Video_Title",
        "YouTube_URL","Transcript_Path","Family_List","Tool_List",
        "Dominant_Tactic","Platform_Signal","Family_Count"
    ]

    X_know = df_know.drop(columns=drop_cols, errors="ignore")\
        .select_dtypes(include="number")\
        .to_numpy(dtype=np.float32)

    # -----------------------------
    # Instantiate agents
    # -----------------------------

    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    know_agent = KnowledgeAgent(T=10)

    BATCH = 512  # GPU batch size for score collection

    # -----------------------------
    # Collect semantic scores (batched)
    # -----------------------------
    print(f"Scoring semantic ({len(X_sem):,} samples)...")
    sem_inputs = [{"embedding": v} for v in X_sem]
    sem_outs = []
    for i in range(0, len(sem_inputs), BATCH):
        sem_outs.extend(sem_agent.score_batch(sem_inputs[i:i+BATCH]))
    p_sem = np.array([o.prob   for o in sem_outs], dtype=np.float32)
    s_sem = np.array([o.uncert for o in sem_outs], dtype=np.float32)
    print(f"  done — mean_p={p_sem.mean():.3f}")

    # -----------------------------
    # Collect behavioural scores (batched)
    # -----------------------------
    print(f"Scoring behavioural ({len(X_beh):,} samples)...")
    beh_inputs = [{"features": v} for v in X_beh]
    beh_outs = []
    for i in range(0, len(beh_inputs), BATCH):
        beh_outs.extend(beh_agent.score_batch(beh_inputs[i:i+BATCH]))
    p_beh = np.array([o.prob   for o in beh_outs], dtype=np.float32)
    s_beh = np.array([o.uncert for o in beh_outs], dtype=np.float32)
    print(f"  done — mean_p={p_beh.mean():.3f}")

    # -----------------------------
    # Collect knowledge scores (batched)
    # -----------------------------
    print(f"Scoring knowledge ({len(X_know):,} samples)...")
    know_inputs = [{"features": v} for v in X_know]
    know_outs = []
    for i in range(0, len(know_inputs), BATCH):
        know_outs.extend(know_agent.score_batch(know_inputs[i:i+BATCH]))
    p_know = np.array([o.prob   for o in know_outs], dtype=np.float32)
    s_know = np.array([o.uncert for o in know_outs], dtype=np.float32)
    print(f"  done — mean_p={p_know.mean():.3f}")

    # -----------------------------
    # Build fusion dataset
    # -----------------------------

    X, y = build_joint_score_dataset(
        p_sem, s_sem, y_sem,
        p_beh, s_beh, y_beh,
        p_know, s_know, y_know
    )

    print("Fusion training set:", X.shape)

    # -----------------------------
    # Train fusion model
    # -----------------------------

    cfg = DecisionTrainConfig(
        use_uncertainties=True,
        ctgan_samples=2000,
        ctgan_epochs=50,
        hidden=32,
        p_drop=0.2,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        early_stop_patience=6,
        valid_size=0.2,
        temperature_scale=True,
    )

    trainer = DecisionTrainer(cfg)

    trainer.fit(X, y)

    ckpt_path = Path("checkpoints/decision_fuser_3agent.json")

    trainer.save(str(ckpt_path))

    print(f"Saved 3-agent decision fusion model → {ckpt_path}")


def build_joint_score_dataset_2agent(p_sem, s_sem, y_sem, p_beh, s_beh, y_beh):
    """
    2-agent fusion dataset, rows [p_sem, p_beh, σ_sem, σ_beh], built exactly as the
    original 2-agent trainer (commented out at the top of this file): each
    semantic row is paired with a same-label behavioural row and vice versa.
    """
    rng = np.random.default_rng(42)
    y_sem_int = y_sem.astype(int)
    y_beh_int = y_beh.astype(int)
    idx_sem = {0: np.where(y_sem_int == 0)[0], 1: np.where(y_sem_int == 1)[0]}
    idx_beh = {0: np.where(y_beh_int == 0)[0], 1: np.where(y_beh_int == 1)[0]}

    def sample_from(indices, max_len):
        if len(indices) > 0:
            return int(rng.choice(indices))
        return int(rng.integers(0, max_len))

    X_rows, y_rows = [], []
    n_sem, n_beh = len(p_sem), len(p_beh)
    for i in range(n_sem):
        k = sample_from(idx_beh[int(y_sem_int[i])], n_beh)
        X_rows.append([p_sem[i], p_beh[k], s_sem[i], s_beh[k]])
        y_rows.append(y_sem[i])
    for j in range(n_beh):
        k = sample_from(idx_sem[int(y_beh_int[j])], n_sem)
        X_rows.append([p_sem[k], p_beh[j], s_sem[k], s_beh[j]])
        y_rows.append(y_beh[j])
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def main_2agent():
    """Train checkpoints/decision_fuser.json (2-agent: semantic + behavioural)."""
    df_sem = pd.read_parquet(SEM_PARQUET)
    df_beh = pd.read_parquet(BEH_PARQUET_F)
    y_sem = df_sem["label"].astype(np.float32).values
    y_beh = df_beh["label"].astype(np.float32).values
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

    sem_agent = SemanticAgent(T=10)
    beh_agent = BehavioralAgent(T=10)
    BATCH = 512

    print(f"Scoring semantic ({len(X_sem):,} samples)...")
    sem_inputs = [{"embedding": v} for v in X_sem]
    sem_outs = []
    for i in range(0, len(sem_inputs), BATCH):
        sem_outs.extend(sem_agent.score_batch(sem_inputs[i:i+BATCH]))
    p_sem = np.array([o.prob for o in sem_outs], dtype=np.float32)
    s_sem = np.array([o.uncert for o in sem_outs], dtype=np.float32)

    print(f"Scoring behavioural ({len(X_beh):,} samples)...")
    beh_inputs = [{"features": v} for v in X_beh]
    beh_outs = []
    for i in range(0, len(beh_inputs), BATCH):
        beh_outs.extend(beh_agent.score_batch(beh_inputs[i:i+BATCH]))
    p_beh = np.array([o.prob for o in beh_outs], dtype=np.float32)
    s_beh = np.array([o.uncert for o in beh_outs], dtype=np.float32)

    X, y = build_joint_score_dataset_2agent(p_sem, s_sem, y_sem, p_beh, s_beh, y_beh)
    print("Fusion training set:", X.shape)

    cfg = DecisionTrainConfig(
        use_uncertainties=True,   # in_dim = 4 (p_sem, p_beh, σ_sem, σ_beh)
        ctgan_samples=2000,
        ctgan_epochs=50,
        hidden=32,
        p_drop=0.2,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        early_stop_patience=6,
        valid_size=0.2,
        temperature_scale=True,
        n_agents=2,
    )
    trainer = DecisionTrainer(cfg)
    trainer.fit(X, y)
    ckpt_path = Path("checkpoints/decision_fuser.json")
    trainer.save(str(ckpt_path))
    print(f"Saved 2-agent decision fusion model → {ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Decision MLP fusion head")
    parser.add_argument("--pipeline", type=int, choices=[2, 3], default=3,
                        help="3: decision_fuser_3agent.json (default); 2: decision_fuser.json")
    args = parser.parse_args()
    main_2agent() if args.pipeline == 2 else main()
