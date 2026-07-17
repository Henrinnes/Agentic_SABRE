# eval_knowledge.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from Agentic_SABRE.agents.knowledge import KnowledgeAgent

if __name__ == "__main__":

    # Load knowledge feature dataset
    df = pd.read_csv(
        r"/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_718.csv"
    )

    # Match the target used during training
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) |
        (df["Tool_Total_Mentions"] > 2)
    ).astype(int)

    # Same columns used during training
    drop_cols = [
        "Video_ID",
        "Channel_ID",
        "Channel_Name",
        "Video_Title",
        "YouTube_URL",
        "Transcript_Path",
        "Family_List",
        "Tool_List",
        "Dominant_Tactic",
        "Platform_Signal",
        "Family_Count",
        "Ransomware_Context",
    ]

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