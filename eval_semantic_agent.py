# eval_semantic_agent.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from Agentic_SABRE.agents.semantic import SemanticAgent

if __name__ == "__main__":
    # 1) load dataset
    df_sem = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet"
    )
    X = np.stack(df_sem["embedding"].values).astype(np.float32)
    y = df_sem["label"].astype(int).values

    # 2) load agent (smaller T for speed)
    agent = SemanticAgent(T=10)

    probs = []
    for v in X:
        out = agent.score({"embedding": v})
        probs.append(out.prob)
    probs = np.array(probs)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    print({"semantic_accuracy": acc, "semantic_auc": auc})
