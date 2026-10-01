# eval_knowledge.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from Agentic_SABRE.agents.knowledge import KnowledgeAgent
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS, RANSKNOW_CSV

if __name__ == "__main__":

    # Load knowledge feature dataset
    df = pd.read_csv(
        RANSKNOW_CSV
    )

    # Match the target used during training
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) |
        (df["Tool_Total_Mentions"] > 2)
    ).astype(int)

    # Restrict to the held-out test split saved by train_knowledge_agent.py
    # so this evaluates generalisation rather than partly in-sample fit.
    test_indices_path = Path("checkpoints/knowledge_test_indices.json")
    if not test_indices_path.exists():
        raise FileNotFoundError(
            f"{test_indices_path} not found -- run train_knowledge_agent.py "
            "first so this script can evaluate on the same held-out test "
            "split rather than the full train+val+test dataset."
        )
    test_indices = json.loads(test_indices_path.read_text())
    df = df.loc[test_indices]
    print(f"Evaluating on held-out test split: {len(df)} rows")

    # Same columns used during training
    drop_cols = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py

    X = (
        df.drop(columns=drop_cols, errors="ignore")
          .select_dtypes(include="number")
          .to_numpy(dtype=np.float32)
    )

    # target label
    y = df["Ransomware_Context"].astype(int).values

    # Load Knowledge Agent
    agent = KnowledgeAgent(
        checkpoint_path="checkpoints/knowledge_cnn.pt",
        metadata_path="checkpoints/metadata.json",
        T=10,
    )

    probs = []
    uncerts = []

    for v in X:
        out = agent.score({"features": v})
        probs.append(out.prob)
        uncerts.append(out.uncert)

    probs = np.array(probs)
    uncerts = np.array(uncerts)

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    print({
        "knowledge_accuracy": acc,
        "knowledge_auc": auc,
        "mean_uncertainty": float(np.mean(uncerts))
    })

    print("\nClassification report:")
    print(classification_report(y, preds))

    print("High uncertainty samples:", int(np.sum(uncerts > 0.25)))
    print("Low uncertainty samples:", int(np.sum(uncerts < 0.05)))