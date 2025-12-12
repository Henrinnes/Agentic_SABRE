# plot_triage_scatter.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.orchestrator import Orchestrator

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
    # choose which policy to visualise:
    # policy_file = "checkpoints/policy_auto.json"           # safety-optimal
    policy_file = "checkpoints/policy_low_escalation.json"   # low-escalation

    policy = json.loads(Path(policy_file).read_text())

    df_sem = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet"
    )
    df_beh = pd.read_parquet(
        r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet"
    )

    y_sem = df_sem["label"].astype(int).values
    y_beh = df_beh["label"].astype(int).values
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    X_beh = df_beh[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)

    orch = Orchestrator(
        τ=policy["tau"],
        κ=policy["kappa"],
        τ_high=policy["tau_high"],
        κ_low=policy["kappa_low"],
        decision_artifact="checkpoints/decision_fuser.json",
    )

    # collect a subset for plotting to avoid millions of points
    n_sem = min(10000, len(X_sem))
    n_beh = min(10000, len(X_beh))

    p_list = []
    sigma_list = []
    dec_list = []
    y_list = []

    # RDset-like pairing: semantic samples paired with random behavioural of same label
    rng = np.random.default_rng(0)
    idx_beh_by_label = {
        0: np.where(y_beh == 0)[0],
        1: np.where(y_beh == 1)[0],
    }

    for i in range(n_sem):
        yi = y_sem[i]
        cand = idx_beh_by_label[yi]
        if len(cand) == 0:
            j = int(rng.integers(0, len(X_beh)))
        else:
            j = int(rng.choice(cand))

        result = orch.handle(
            {"embedding": X_sem[i]}, {"features": X_beh[j]}, context=None
        )
        p_list.append(result["score_fused"])
        sigma_list.append(result["uncert_max"])
        dec_list.append(result["decision"])
        y_list.append(int(yi))

    p = np.asarray(p_list)
    s = np.asarray(sigma_list)
    y = np.asarray(y_list)

    # map decision to colours/markers
    color_map = {"AUTO_CONTAIN": "red", "ESCALATE": "orange", "ALLOW": "green"}
    marker_map = {0: "o", 1: "x"}  # benign vs ransomware

    plt.figure(figsize=(8, 6))
    for decision in ["AUTO_CONTAIN", "ESCALATE", "ALLOW"]:
        mask = np.array(dec_list) == decision
        plt.scatter(
            p[mask], s[mask],
            c=color_map[decision],
            alpha=0.4,
            label=decision,
        )

    # draw policy lines
    τ = policy["tau"]
    κ = policy["kappa"]
    τ_high = policy["tau_high"]
    κ_low = policy["kappa_low"]

    plt.axvline(τ, linestyle="--", linewidth=1)
    plt.axvline(τ_high, linestyle="--", linewidth=1)
    plt.axhline(κ, linestyle="--", linewidth=1)
    plt.axhline(κ_low, linestyle="--", linewidth=1)

    plt.xlabel(r"Fused risk $\hat{p}$")
    plt.ylabel(r"Max uncertainty $\sigma_{\max}$")
    plt.title("Triage policy in $(\hat{p}, \sigma_{\max})$ space")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
