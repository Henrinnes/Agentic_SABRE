# behavioural_feature_importance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from Agentic_SABRE.agents.behavioural import BehavioralAgent

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def main():
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet"
    )
    X = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y = df_beh["label"].astype(int).values

    beh_agent = BehavioralAgent(T=10)

    # baseline probs
    print("Computing baseline behavioural probabilities...")
    p_base = []
    for v in X:
        out = beh_agent.score({"features": v})
        p_base.append(out.prob)
    p_base = np.asarray(p_base, dtype=np.float32)
    auc_base = roc_auc_score(y, p_base)
    print("Baseline AUC:", auc_base)

    rng = np.random.default_rng(0)
    importances = []

    for col_idx, col_name in enumerate(BEHAVIOR_COLUMNS):
        X_perturb = X.copy()
        # shuffle this column
        rng.shuffle(X_perturb[:, col_idx])

        p_perm = []
        for v in X_perturb:
            out = beh_agent.score({"features": v})
            p_perm.append(out.prob)
        p_perm = np.asarray(p_perm, dtype=np.float32)
        auc_perm = roc_auc_score(y, p_perm)
        drop = auc_base - auc_perm
        importances.append(drop)
        print(f"{col_name}: AUC={auc_perm:.4f}, drop={drop:.4f}")

    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(BEHAVIOR_COLUMNS)), importances[order])
    plt.xticks(range(len(BEHAVIOR_COLUMNS)), [BEHAVIOR_COLUMNS[i] for i in order], rotation=45, ha="right")
    plt.ylabel("AUC drop on permutation")
    plt.title("Behavioural feature importance (permutation)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
