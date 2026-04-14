#eval_fusion_policy_3agent.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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

KNOW_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal", "Family_Count"
]


def build_knowledge_matrix(df_know: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_know = (
        df_know.drop(columns=KNOW_DROP_COLS, errors="ignore")
               .select_dtypes(include="number")
               .to_numpy(dtype=np.float32)
    )

    # Keep the same label definition used in your current knowledge eval/training path
    y_know = (df_know["Family_Count"] > 0).astype(int).values
    return X_know, y_know


def evaluate_dataset(name, X_sem, y_sem, X_beh, y_beh, X_know, y_know, orch):
    fused_probs, max_uncerts, decisions, labels = [], [], [], []

    # Shadow evaluation only: align by truncating to common length
    N = min(len(y_sem), len(y_beh), len(y_know))

    for i in range(N):
        sem_input = {"embedding": X_sem[i]}
        beh_input = {"features": X_beh[i]}
        know_input = {"features": X_know[i]}

        # Use semantic labels as the reference labels for this engineering validation run
        label = int(y_sem[i])

        out = orch.handle(
            x_sem=sem_input,
            x_beh=beh_input,
            x_know=know_input,
            context={"sample_index": i}
        )

        fused_probs.append(float(out["score_fused"]))
        max_uncerts.append(float(out["uncert_max"]))
        decisions.append(out["decision"])
        labels.append(label)

    fused_probs = np.asarray(fused_probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    preds = (fused_probs >= 0.5).astype(int)

    return {
        "name": name,
        "n_samples": int(N),
        "acc": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, fused_probs)),
        "mean_uncert": float(np.mean(max_uncerts)),
        "decision_counts": {
            "AUTO_CONTAIN": int(decisions.count("AUTO_CONTAIN")),
            "ESCALATE": int(decisions.count("ESCALATE")),
            "ALLOW": int(decisions.count("ALLOW")),
        },
        "confusion": confusion_matrix(labels, preds).tolist(),
    }


if __name__ == "__main__":
    meta = json.loads(Path("checkpoints/metadata.json").read_text())

    print("Loaded metadata dims:")
    print({
        "semantic_input_dim": meta.get("semantic_input_dim"),
        "behavioral_input_dim": meta.get("behavioral_input_dim"),
        "knowledge_input_dim": meta.get("knowledge_input_dim"),
    })

    # Load datasets
    df_sem = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet"
    )
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet"
    )
    df_know = pd.read_csv(
        r"C:\Users\henri\SampleDatasets\Dataset_RansKnow\outputs\Knowledge_Agent_Features_307.csv"
    )

    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(int).values

    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y_beh = df_beh["label"].astype(int).values

    X_know, y_know = build_knowledge_matrix(df_know)

    policy = json.loads(Path("checkpoints/policy_low_escalation.json").read_text())

    orch = Orchestrator(
        τ=policy["tau"],
        κ=policy["kappa"],
        τ_high=policy["tau_high"],
        κ_low=policy["kappa_low"],
        decision_artifact="checkpoints/decision_fuser_3agent.json",
    )

    results = evaluate_dataset(
        name="Fusion_3Agent_ShadowEval",
        X_sem=X_sem,
        y_sem=y_sem,
        X_beh=X_beh,
        y_beh=y_beh,
        X_know=X_know,
        y_know=y_know,
        orch=orch,
    )

    print("=== 3-Agent Fusion Evaluation ===")
    print(json.dumps(results, indent=2))