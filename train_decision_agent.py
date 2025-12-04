# train_decision_agent.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.agents.decision import DecisionTrainer, DecisionTrainConfig

def main():
    # 1) Load datasets (same ones used for CNN training)
    df_sem = pd.read_parquet(r'C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet')
    df_beh = pd.read_parquet(r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet")

    # assume labels aligned / same length; if not, you’ll need your pairing logic
    y = df_sem["label"].astype(np.float32).values

    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    behavior_columns = [
        'avg_read_throughput', 'avg_write_throughput',
        'read_LBA_variance', 'write_LBA_variance',
        'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
        'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
        'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
        'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
        'mem_readwrite_count', 'mem_exec_count'
    ]
    X_beh = df_beh[behavior_columns].to_numpy(dtype=np.float32)

    sem_agent = SemanticAgent()
    beh_agent = BehavioralAgent()

    p_sem, s_sem = [], []
    p_beh, s_beh = [], []

    for v in X_sem:
        out = sem_agent.score({"embedding": v})
        p_sem.append(out.prob)
        s_sem.append(out.uncert)

    for v in X_beh:
        out = beh_agent.score({"features": v})
        p_beh.append(out.prob)
        s_beh.append(out.uncert)

    p_sem = np.asarray(p_sem, dtype=np.float32)
    p_beh = np.asarray(p_beh, dtype=np.float32)
    s_sem = np.asarray(s_sem, dtype=np.float32)
    s_beh = np.asarray(s_beh, dtype=np.float32)

    # X has columns [p_sem, p_beh, σ_sem, σ_beh]
    X = np.stack([p_sem, p_beh, s_sem, s_beh], axis=1)

    cfg = DecisionTrainConfig(
        use_uncertainties=True,
        ctgan_samples=2000,   # tweak
        ctgan_epochs=50,
        hidden=32,
        p_drop=0.2,
        lr=1e-3,
        batch_size=128,
        max_epochs=50,
        early_stop_patience=6,
        valid_size=0.2,
        temperature_scale=True,
    )

    trainer = DecisionTrainer(cfg)
    trainer.fit(X, y)
    ckpt_path = Path("checkpoints/decision_fuser.json")
    trainer.save(str(ckpt_path))
    print(f"Saved decision fusion artifact to {ckpt_path}")

if __name__ == "__main__":
    main()
