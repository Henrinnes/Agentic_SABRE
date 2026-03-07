import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from Agentic_SABRE.agents.knowledge import KnowledgeAgent

if __name__ == "__main__":

    # Load knowledge feature dataset
    df = pd.read_csv(
        r"/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv"
    )

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
        "Family_Count"
    ]

    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number").to_numpy(dtype=np.float32)

    # target label
    y = (df["Family_Count"] > 0).astype(int).values

    # Load Knowledge Agent
    agent = KnowledgeAgent(T=10)

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