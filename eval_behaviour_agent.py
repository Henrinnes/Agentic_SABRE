# eval_behavioral_agent.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from Agentic_SABRE.agents.behavioural import BehavioralAgent

if __name__ == "__main__":
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet"
    )
    behavior_columns = [
        'avg_read_throughput', 'avg_write_throughput',
        'read_LBA_variance', 'write_LBA_variance',
        'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
        'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
        'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
        'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
        'mem_readwrite_count', 'mem_exec_count'
    ]
    X = df_beh[behavior_columns].to_numpy(dtype=np.float32)
    y = df_beh["label"].astype(int).values

    agent = BehavioralAgent(T=10)

    probs = []
    for v in X:
        out = agent.score({"features": v})
        probs.append(out.prob)
    probs = np.array(probs)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)

    print({"behavioral_accuracy": acc, "behavioral_auc": auc})
