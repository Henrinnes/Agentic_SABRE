# eval_fusion_policy.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from Agentic_SABRE.orchestrator import Orchestrator, load_auto_policy

behavior_columns = [
    'avg_read_throughput', 'avg_write_throughput',
    'read_LBA_variance', 'write_LBA_variance',
    'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
    'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
    'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
    'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
    'mem_readwrite_count', 'mem_exec_count'
]

def evaluate_dataset(name, X_sem, y_sem, X_beh, y_beh, orch):
    fused_probs, max_uncerts, decisions, labels = [], [], [], []

    N = max(len(y_sem), len(y_beh))
    for i in range(N):
        sem_input = {"embedding": X_sem[i]} if i < len(X_sem) else {"embedding": np.zeros_like(X_sem[0])}
        beh_input = {"features":  X_beh[i]} if i < len(X_beh) else {"features":  np.zeros_like(X_beh[0])}
        label = y_sem[i] if i < len(y_sem) else y_beh[i - len(y_sem)]

        out = orch.handle(sem_input, beh_input)

        fused_probs.append(out["score_fused"])
        max_uncerts.append(out["uncert_max"])
        decisions.append(out["decision"])
        labels.append(label)

    preds = (np.array(fused_probs) >= 0.5).astype(int)

    return {
        "name": name,
        "acc": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, fused_probs),
        "decision_counts": {
            "AUTO_CONTAIN": decisions.count("AUTO_CONTAIN"),
            "ESCALATE": decisions.count("ESCALATE"),
            "ALLOW": decisions.count("ALLOW"),
        },
        "confusion": confusion_matrix(labels, preds).tolist(),
    }


if __name__ == "__main__":
    # Load the saved metadata (dims)
    meta = json.loads(Path("checkpoints/metadata.json").read_text())
    d_sem = meta["semantic_input_dim"]
    d_beh = meta["behavioral_input_dim"]

    # Load datasets
    df_sem = pd.read_parquet(r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet")
    df_beh = pd.read_parquet(r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet")

    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(int).values

    X_beh = df_beh[behavior_columns].to_numpy(dtype=np.float32)
    y_beh = df_beh["label"].astype(int).values

    # orch = Orchestrator(
    #     τ=0.80, κ=0.15,
    #     τ_high=0.95, κ_low=0.08,
    #     decision_artifact="checkpoints/decision_fuser.json"
    # )
    
    # load auto-tuned policy
    # policy = json.loads(Path("checkpoints/policy_auto.json").read_text())
    policy = json.loads(Path("checkpoints/policy_low_escalation.json").read_text())


    orch = Orchestrator(
        τ=policy["tau"],
        κ=policy["kappa"],
        τ_high=policy["tau_high"],
        κ_low=policy["kappa_low"],
        decision_artifact="checkpoints/decision_fuser.json",
    )


    rd_results = evaluate_dataset("RDset", X_sem, y_sem, X_beh[:len(y_sem)], y_sem, orch)
    beh_results = evaluate_dataset("RanSMAP", X_sem[:len(y_beh)], y_beh, X_beh, y_beh, orch)

    print("=== Fusion Evaluation ===")
    print(rd_results)
    print(beh_results)