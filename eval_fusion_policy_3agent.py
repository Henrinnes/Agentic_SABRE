# eval_fusion_policy_3agent.py
"""
Evaluation script for the 3-agent (Semantic + Behavioural + Knowledge) fusion pipeline.

Dataset note
------------
RDset    : PE static-analysis samples  → semantic embeddings + labels
RanSMAP  : Runtime-monitoring samples  → behavioural features + labels
RansKnow : MITRE ATT&CK / threat intel → knowledge features + labels

All three are INDEPENDENT datasets — no sample appears in more than one.
A joint fusion evaluation across all three modalities from the same executable
is not possible without a unified dataset.  Each section below is clearly
labelled with what is and is not being measured.
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
)

from Agentic_SABRE.orchestrator_3agent import Orchestrator
from Agentic_SABRE.agents.semantic    import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.agents.knowledge   import KnowledgeAgent

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOW_DROP_COLS = [
    "Video_ID", "Channel_ID", "Channel_Name", "Video_Title",
    "YouTube_URL", "Transcript_Path", "Family_List", "Tool_List",
    "Dominant_Tactic", "Platform_Signal", "Family_Count", "Ransomware_Context",
    "label",
]

BATCH_SIZE = 256


def stratified_sample(df, n_per_class, seed=42):
    parts = [
        df[df.label == lbl].sample(min(n_per_class, (df.label == lbl).sum()), random_state=seed)
        for lbl in [0, 1]
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def stratified_sample_by(df, label_col, n_per_class, seed=42):
    parts = [
        df[df[label_col] == lbl].sample(
            min(n_per_class, (df[label_col] == lbl).sum()), random_state=seed
        )
        for lbl in [0, 1]
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def print_metrics(name, labels, probs, elapsed):
    preds = (np.array(probs) >= 0.5).astype(int)
    N = len(labels)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples  : {N}  |  Time: {elapsed:.2f}s  ({N/elapsed:.0f} samples/s)")
    print(f"  Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(labels, probs):.4f}")
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion matrix:")
    print(f"              Pred Benign  Pred Ransom")
    print(f"  True Benign    {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  True Ransom    {cm[1,0]:5d}       {cm[1,1]:5d}")
    print(classification_report(labels, preds, target_names=["Benign", "Ransomware"], digits=4))


def print_fusion_metrics(name, labels, probs, uncerts, decisions, elapsed, note=""):
    preds = (np.array(probs) >= 0.5).astype(int)
    N = len(labels)
    print(f"\n{'='*60}")
    print(f"  {name}")
    if note:
        print(f"  NOTE: {note}")
    print(f"{'='*60}")
    print(f"  Samples  : {N}  |  Time: {elapsed:.2f}s  ({N/elapsed:.0f} samples/s)")
    print(f"  Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(labels, probs):.4f}")
    print(f"  Avg fused prob : {np.mean(probs):.4f}   Avg max uncert: {np.mean(uncerts):.4f}")
    print(f"  Triage — AUTO_CONTAIN: {decisions.count('AUTO_CONTAIN')}  "
          f"ESCALATE: {decisions.count('ESCALATE')}  ALLOW: {decisions.count('ALLOW')}")
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion matrix:")
    print(f"              Pred Benign  Pred Ransom")
    print(f"  True Benign    {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  True Ransom    {cm[1,0]:5d}       {cm[1,1]:5d}")
    print(classification_report(labels, preds, target_names=["Benign", "Ransomware"], digits=4))


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Load data
    # ------------------------------------------------------------------ #
    print("Loading datasets...")
    df_sem = pd.read_parquet(
        r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet")
    df_beh = pd.read_parquet(
        r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet")
    df_know_raw = pd.read_csv(
        r"/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv")

    # Build knowledge label (must match training)
    df_know_raw["label"] = (
        (df_know_raw["Tactic_Total_Mentions"] > 10) |
        (df_know_raw["Tool_Total_Mentions"] > 2)
    ).astype(int)

    print(f"  RDset    (semantic)   : {len(df_sem):,} rows  "
          f"(ransom={df_sem.label.sum():,}  benign={(df_sem.label==0).sum():,})")
    print(f"  RanSMAP  (behavioural): {len(df_beh):,} rows  "
          f"(ransom={df_beh.label.sum():,}  benign={(df_beh.label==0).sum():,})")
    print(f"  RansKnow (knowledge)  : {len(df_know_raw):,} rows  "
          f"(ransom={df_know_raw.label.sum():,}  benign={(df_know_raw.label==0).sum():,})")

    # Stratified samples
    sem_s  = stratified_sample(df_sem, 2000)
    beh_s  = stratified_sample(df_beh, 1000)
    know_s = stratified_sample_by(df_know_raw, "label", 100)   # small dataset

    N_sem  = len(sem_s)
    N_beh  = len(beh_s)
    N_know = len(know_s)
    print(f"\n  Sampled RDset   : {N_sem}  ({(sem_s.label==1).sum()} ransom / {(sem_s.label==0).sum()} benign)")
    print(f"  Sampled RanSMAP : {N_beh}  ({(beh_s.label==1).sum()} ransom / {(beh_s.label==0).sum()} benign)")
    print(f"  Sampled RansKnow: {N_know}  ({(know_s.label==1).sum()} ransom / {(know_s.label==0).sum()} benign)")

    X_sem  = np.stack(sem_s["embedding"].values).astype(np.float32)
    y_sem  = sem_s["label"].astype(int).values
    X_beh  = beh_s[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y_beh  = beh_s["label"].astype(int).values
    X_know = (
        know_s.drop(columns=KNOW_DROP_COLS, errors="ignore")
              .select_dtypes(include="number")
              .to_numpy(dtype=np.float32)
    )
    y_know = know_s["label"].astype(int).values

    # ------------------------------------------------------------------ #
    #  Load policy + orchestrator
    # ------------------------------------------------------------------ #
    policy = json.loads(Path("checkpoints/policy_low_escalation.json").read_text())
    print(f"\nPolicy (low-escalation): tau={policy['tau']:.4f}  kappa={policy['kappa']:.4f}  "
          f"tau_high={policy['tau_high']:.4f}  kappa_low={policy['kappa_low']:.4f}")

    orch = Orchestrator(
        **{"τ": policy["tau"], "κ": policy["kappa"],
           "τ_high": policy["tau_high"], "κ_low": policy["kappa_low"]},
        decision_artifact="checkpoints/decision_fuser_3agent.json",
    )

    # Warm-up GPU
    orch.handle_batch(
        [{"embedding": X_sem[i]} for i in range(32)],
        [{"features":  X_beh[i]} for i in range(32)],
        [{"features":  X_know[i % N_know]} for i in range(32)],
    )

    # ================================================================== #
    #  SECTION 1 — Per-agent evaluations (each on its native dataset)
    # ================================================================== #
    print("\n\n" + "#"*60)
    print("# SECTION 1 — Per-Agent Evaluations (native datasets)")
    print("# Each agent evaluated independently on the data it was trained for.")
    print("#"*60)

    # 1a. Semantic
    print("\n[1a] Semantic agent on RDset (static PE features)")
    inputs_sem = [{"embedding": X_sem[i]} for i in range(N_sem)]
    t0 = time.time()
    outputs = []
    for start in range(0, N_sem, BATCH_SIZE):
        outputs.extend(orch.sem.score_batch(inputs_sem[start:start+BATCH_SIZE]))
    elapsed = time.time() - t0
    print_metrics("Semantic Agent — RDset", y_sem, [o.prob for o in outputs], elapsed)

    # 1b. Behavioural
    print("\n[1b] Behavioural agent on RanSMAP (runtime telemetry features)")
    inputs_beh = [{"features": X_beh[i]} for i in range(N_beh)]
    t0 = time.time()
    outputs = []
    for start in range(0, N_beh, BATCH_SIZE):
        outputs.extend(orch.beh.score_batch(inputs_beh[start:start+BATCH_SIZE]))
    elapsed = time.time() - t0
    print_metrics("Behavioural Agent — RanSMAP", y_beh, [o.prob for o in outputs], elapsed)

    # 1c. Knowledge
    print("\n[1c] Knowledge agent on RansKnow (MITRE ATT&CK features)")
    inputs_know = [{"features": X_know[i]} for i in range(N_know)]
    t0 = time.time()
    outputs = []
    for start in range(0, N_know, BATCH_SIZE):
        outputs.extend(orch.know.score_batch(inputs_know[start:start+BATCH_SIZE]))
    elapsed = time.time() - t0
    print_metrics("Knowledge Agent — RansKnow", y_know, [o.prob for o in outputs], elapsed)

    # ================================================================== #
    #  SECTION 2 — 3-agent fusion pipeline evaluation on RDset
    #
    #  IMPORTANT: behavioural inputs are from RanSMAP and knowledge inputs
    #  are from RansKnow — all three datasets are independent. The semantic
    #  agent dominates the fusion decision; behavioural and knowledge
    #  features act as cross-dataset noise from the fusion model's perspective.
    #
    #  Interpretation: metrics reflect semantic agent performance.
    #  A valid fusion eval requires a unified dataset with all three
    #  modalities from the same sample.
    # ================================================================== #
    print("\n\n" + "#"*60)
    print("# SECTION 2 — 3-Agent Fusion Pipeline Evaluation on RDset")
    print("# Semantic: RDset (matched).  Behavioural: RanSMAP (CROSS-DATASET).")
    print("# Knowledge: RansKnow (CROSS-DATASET).")
    print("# Metrics reflect semantic agent dominance — NOT true fusion performance.")
    print("#"*60)

    N_fuse = min(N_sem, N_beh, N_know)
    x_sems_f  = [{"embedding": X_sem[i]}            for i in range(N_fuse)]
    x_behs_f  = [{"features":  X_beh[i]}             for i in range(N_fuse)]
    x_knows_f = [{"features":  X_know[i % N_know]}   for i in range(N_fuse)]
    labels_f  = y_sem[:N_fuse]

    fused_probs, fused_uncerts, fused_decisions = [], [], []
    t0 = time.time()
    for start in range(0, N_fuse, BATCH_SIZE):
        for out in orch.handle_batch(
            x_sems_f[start:start+BATCH_SIZE],
            x_behs_f[start:start+BATCH_SIZE],
            x_knows_f[start:start+BATCH_SIZE],
        ):
            fused_probs.append(out["score_fused"])
            fused_uncerts.append(out["uncert_max"])
            fused_decisions.append(out["decision"])
    elapsed = time.time() - t0

    print_fusion_metrics(
        "3-Agent Fusion Pipeline — RDset labels, cross-dataset beh + know input",
        labels_f, fused_probs, fused_uncerts, fused_decisions, elapsed,
        note="Behavioural (RanSMAP) and Knowledge (RansKnow) inputs are from different samples. "
             "Accuracy reflects semantic agent, not true fusion.",
    )

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print("  Semantic agent   (RDset)    : valid standalone eval")
    print("  Behavioural agent (RanSMAP) : valid standalone eval")
    print("  Knowledge agent  (RansKnow) : valid standalone eval")
    print("  Fusion pipeline             : semantic-driven; beh + know are cross-dataset noise")
    print("  For true fusion eval        : provide a unified dataset with all three modalities")
    print("="*60)
