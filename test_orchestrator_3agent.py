import os
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS, RANSKNOW_CSV
#Agentic_SABRE\test_orchestrator_3agent.py
# import json
# from pathlib import Path
# import numpy as np
# import pandas as pd

# from Agentic_SABRE.orchestrator_3agent import Orchestrator

# BEHAVIOR_COLUMNS = [
#     "avg_read_throughput", "avg_write_throughput",
#     "read_LBA_variance", "write_LBA_variance",
#     "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
#     "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
#     "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
#     "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
#     "mem_readwrite_count", "mem_exec_count",
# ]

# KNOW_DROP_COLS = [
#     "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
#     "YouTube_URL", "Transcript_Path", "Family_List", "Tool_List",
#     "Dominant_Tactic", "Platform_Signal", "Family_Count"
# ]

# # sample semantic row
# df_sem = pd.read_parquet(
#     r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
# )

# # sample behavioural row
# df_beh = pd.read_parquet(
#     r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
# )

# # sample knowledge row
# df_know = pd.read_csv(
#     r"/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv"
# )

# sem_vec = np.asarray(df_sem.iloc[0]["embedding"], dtype=np.float32)
# beh_vec = df_beh[BEHAVIOR_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
# know_vec = (
#     df_know.drop(columns=KNOW_DROP_COLS, errors="ignore")
#            .select_dtypes(include="number")
#            .iloc[0]
#            .to_numpy(dtype=np.float32)
# )

# orch = Orchestrator(
#     decision_artifact="checkpoints/decision_fuser_3agent.json"
# )

# out = orch.handle(
#     x_sem={"embedding": sem_vec},
#     x_beh={"features": beh_vec},
#     x_know={"features": know_vec},
#     context={"host": "test-host", "pid": 1234}
# )

# print(json.dumps(out, indent=2))


import json
from pathlib import Path
import numpy as np
import pandas as pd

from Agentic_SABRE.orchestrator_3agent import Orchestrator

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOW_DROP_COLS = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py

_DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
SEM_PARQUET = str(_DATA_DIR / "RDset" / "semantic_dataset_with_embeddings.parquet")
BEH_PARQUET_F = os.environ.get("SABRE_BEH_PARQUET", str(_DATA_DIR / "ransmap_extracted" / "combined_i3-gen12_datasetF.parquet"))
df_sem = pd.read_parquet(
    SEM_PARQUET
)

df_beh = pd.read_parquet(
    BEH_PARQUET_F
)

df_know = pd.read_csv(
    RANSKNOW_CSV
)

sem_vec = np.asarray(df_sem.iloc[0]["embedding"], dtype=np.float32)
beh_vec = df_beh[BEHAVIOR_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
know_vec = (
    df_know.drop(columns=KNOW_DROP_COLS, errors="ignore")
           .select_dtypes(include="number")
           .iloc[0]
           .to_numpy(dtype=np.float32)
)

orch = Orchestrator(
    decision_artifact="checkpoints/decision_fuser_3agent.json"
)

out = orch.handle(
    x_sem={"embedding": sem_vec},
    x_beh={"features": beh_vec},
    x_know={"features": know_vec},
    context={"host": "test-host", "pid": 1234}
)

def to_jsonable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

print(json.dumps(out, indent=2, default=to_jsonable))