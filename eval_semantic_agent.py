# eval_semantic_agent.py
#
# Scores the Semantic Agent on RDset. The headline numbers are on the 20%
# held-out split that train_cnn_agents.py does not train on (stratified,
# random_state=42); the all-rows numbers include the training rows and are
# reported only as an in-sample reference.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from Agentic_SABRE.agents.semantic import SemanticAgent

DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
# Built by data_pipeline/rdset_semantic.py (all-MiniLM-L6-v2, Name/md5 dropped first).
RDSET_PARQUET = DATA_DIR / "RDset" / "semantic_dataset_with_embeddings.parquet"

if __name__ == "__main__":
    # 1) load dataset
    df_sem = pd.read_parquet(RDSET_PARQUET, columns=["embedding", "label"])
    X = np.stack(df_sem["embedding"].values).astype(np.float32)
    y = df_sem["label"].astype(int).values

    # 2) load agent (smaller T for speed)
    agent = SemanticAgent(T=10)

    probs = []
    for v in X:
        out = agent.score({"embedding": v})
        probs.append(out.prob)
    probs = np.array(probs)

    _, test_idx = train_test_split(np.arange(len(y)), stratify=y, test_size=0.2, random_state=42)
    results = {}
    for name, idx in (("held_out", test_idx), ("all_rows_in_sample", np.arange(len(y)))):
        preds = (probs[idx] >= 0.5).astype(int)
        results[f"semantic_accuracy_{name}"] = float(accuracy_score(y[idx], preds))
        results[f"semantic_auc_{name}"] = float(roc_auc_score(y[idx], probs[idx]))

    print(json.dumps(results, indent=2))
