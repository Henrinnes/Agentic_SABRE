# eval_behaviour_agent.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, average_precision_score,
)

from Agentic_SABRE.agents.behavioural import BehavioralAgent

BEHAVIOR_COLUMNS = [
    'avg_read_throughput', 'avg_write_throughput',
    'read_LBA_variance', 'write_LBA_variance',
    'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
    'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
    'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
    'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
    'mem_readwrite_count', 'mem_exec_count',
]

if __name__ == "__main__":
    df_beh = pd.read_parquet(
        r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
    )

    X = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y = df_beh["label"].astype(int).values

    # Hold out the same 20% stratified test split used during training (seed=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Test set: {len(y_test)} samples | "
          f"ransomware={y_test.sum()} ({y_test.mean()*100:.1f}%) | "
          f"benign={len(y_test)-y_test.sum()}")

    agent = BehavioralAgent(T=10)
    # scaler is applied automatically inside agent.score_batch if behavioral_scaler.joblib exists

    probs = agent.score_batch(
        [{"features": v} for v in X_test]
    )
    probs = np.array([o.prob for o in probs])

    preds = (probs >= 0.5).astype(int)

    print({
        "behavioral_accuracy":  round(accuracy_score(y_test, preds), 4),
        "behavioral_precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "behavioral_recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "behavioral_f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "behavioral_roc_auc":   round(roc_auc_score(y_test, probs), 4),
        "behavioral_pr_auc":    round(average_precision_score(y_test, probs), 4),
    })
