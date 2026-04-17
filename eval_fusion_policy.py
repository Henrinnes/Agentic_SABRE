# eval_fusion_policy.py
"""
Evaluation script for the 2-agent (Semantic + Behavioral) fusion pipeline.

Dataset note
------------
RDset   : PE static-analysis samples → semantic embeddings + labels
RanSMAP : Runtime-monitoring samples → behavioral features + labels

These are INDEPENDENT datasets — no sample appears in both.
A joint fusion evaluation (both modalities from the same executable) is not
possible without a unified dataset. Each section below is clearly labelled with
what is and is not being measured.
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from Agentic_SABRE.orchestrator import Orchestrator
from Agentic_SABRE.agents.semantic import SemanticAgent
from Agentic_SABRE.agents.behavioural import BehavioralAgent
from Agentic_SABRE.contracts import AgentOutput

BEHAVIOR_COLUMNS = [
    'avg_read_throughput', 'avg_write_throughput',
    'read_LBA_variance', 'write_LBA_variance',
    'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
    'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
    'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
    'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
    'mem_readwrite_count', 'mem_exec_count'
]

BATCH_SIZE = 256


def stratified_sample(df, n_per_class, seed=42):
    parts = [df[df.label == lbl].sample(min(n_per_class, (df.label == lbl).sum()), random_state=seed)
             for lbl in [0, 1]]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def print_metrics(name, labels, probs, decisions, elapsed):
    preds = (np.array(probs) >= 0.5).astype(int)
    N = len(labels)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples  : {N}  |  Time: {elapsed:.2f}s  ({N/elapsed:.0f} samples/s)")
    print(f"  Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(labels, probs):.4f}")
    print(f"  Avg prob : {np.mean(probs):.4f}   Avg uncert: N/A (single-agent)")
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
    print(f"  RDset   (semantic) : {len(df_sem):,} rows  "
          f"(ransom={df_sem.label.sum():,}  benign={(df_sem.label==0).sum():,})")
    print(f"  RanSMAP (behavioral): {len(df_beh):,} rows  "
          f"(ransom={df_beh.label.sum():,}  benign={(df_beh.label==0).sum():,})")

    # Stratified samples — 2000 per class where available
    sem_s = stratified_sample(df_sem, 2000)
    beh_s = stratified_sample(df_beh, 1000)   # only 1259 benign in RanSMAP
    N_sem = len(sem_s)
    N_beh = len(beh_s)
    print(f"\n  Sampled RDset   : {N_sem} ({(sem_s.label==1).sum()} ransom / {(sem_s.label==0).sum()} benign)")
    print(f"  Sampled RanSMAP : {N_beh} ({(beh_s.label==1).sum()} ransom / {(beh_s.label==0).sum()} benign)")

    X_sem = np.stack(sem_s["embedding"].values).astype(np.float32)
    y_sem = sem_s["label"].astype(int).values
    X_beh = beh_s[BEHAVIOR_COLUMNS].to_numpy(dtype=np.float32)
    y_beh = beh_s["label"].astype(int).values

    # ------------------------------------------------------------------ #
    #  Load policy + orchestrator
    # ------------------------------------------------------------------ #
    policy = json.loads(Path("checkpoints/policy_low_escalation.json").read_text())
    print(f"\nPolicy (low-escalation): tau={policy['tau']:.4f}  kappa={policy['kappa']:.4f}  "
          f"tau_high={policy['tau_high']:.4f}  kappa_low={policy['kappa_low']:.4f}")

    orch = Orchestrator(
        **{'τ': policy['tau'], 'κ': policy['kappa'],
           'τ_high': policy['tau_high'], 'κ_low': policy['kappa_low']},
        decision_artifact="checkpoints/decision_fuser.json",
    )

    # Warm-up GPU
    orch.handle_batch(
        [{"embedding": X_sem[i]} for i in range(32)],
        [{"features":  X_beh[i]} for i in range(32)],
    )

    # ================================================================== #
    #  SECTION 1 — Per-agent evaluations (each on its native dataset)
    #  These are valid standalone evaluations.
    # ================================================================== #
    print("\n\n" + "#"*60)
    print("# SECTION 1 — Per-Agent Evaluations (native datasets)")
    print("# Each agent evaluated independently on the data it was trained for.")
    print("#"*60)

    # --- 1a. Semantic agent on RDset ---
    print("\n[1a] Semantic agent on RDset (static PE features)")
    sem_agent = orch.sem
    inputs_sem = [{"embedding": X_sem[i]} for i in range(N_sem)]
    t0 = time.time()
    outputs = []
    for start in range(0, N_sem, BATCH_SIZE):
        outputs.extend(sem_agent.score_batch(inputs_sem[start:start+BATCH_SIZE]))
    elapsed = time.time() - t0
    probs_sem = [o.prob for o in outputs]
    print_metrics(
        "Semantic Agent — RDset",
        y_sem, probs_sem,
        decisions=[], elapsed=elapsed,
    )

    # --- 1b. Behavioral agent on RanSMAP ---
    print("\n[1b] Behavioral agent on RanSMAP (runtime telemetry features)")
    beh_agent = orch.beh
    inputs_beh = [{"features": X_beh[i]} for i in range(N_beh)]
    t0 = time.time()
    outputs = []
    for start in range(0, N_beh, BATCH_SIZE):
        outputs.extend(beh_agent.score_batch(inputs_beh[start:start+BATCH_SIZE]))
    elapsed = time.time() - t0
    probs_beh = [o.prob for o in outputs]
    print_metrics(
        "Behavioral Agent — RanSMAP",
        y_beh, probs_beh,
        decisions=[], elapsed=elapsed,
    )

    # ================================================================== #
    #  SECTION 2 — Fusion pipeline evaluation on RDset
    #
    #  IMPORTANT: behavioral inputs here are drawn from RanSMAP (a different
    #  dataset). They are NOT matched to the same executables as the semantic
    #  embeddings. The semantic agent dominates the fusion decision; behavioral
    #  features act as noise from the fusion model's perspective.
    #
    #  Interpretation: metrics reflect semantic agent performance, not true
    #  multi-modal fusion. A valid fusion eval requires a unified dataset
    #  with both modalities from the same sample.
    # ================================================================== #
    print("\n\n" + "#"*60)
    print("# SECTION 2 — Fusion Pipeline Evaluation on RDset")
    print("# Semantic: RDset (matched).  Behavioral: RanSMAP (CROSS-DATASET).")
    print("# Metrics reflect semantic agent dominance — NOT true fusion performance.")
    print("#"*60)

    N_fuse = min(N_sem, N_beh)
    x_sems_f = [{"embedding": X_sem[i]} for i in range(N_fuse)]
    x_behs_f = [{"features":  X_beh[i]} for i in range(N_fuse)]
    labels_f  = y_sem[:N_fuse]

    fused_probs, fused_uncerts, fused_decisions = [], [], []
    t0 = time.time()
    for start in range(0, N_fuse, BATCH_SIZE):
        for out in orch.handle_batch(x_sems_f[start:start+BATCH_SIZE], x_behs_f[start:start+BATCH_SIZE]):
            fused_probs.append(out["score_fused"])
            fused_uncerts.append(out["uncert_max"])
            fused_decisions.append(out["decision"])
    elapsed = time.time() - t0

    print_fusion_metrics(
        "Fusion Pipeline — RDset labels, cross-dataset behavioral input",
        labels_f, fused_probs, fused_uncerts, fused_decisions, elapsed,
        note="Behavioral input is from RanSMAP (different samples). "
             "Accuracy reflects semantic agent, not fusion."
    )

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print("  Semantic agent (RDset)     : valid standalone eval")
    print("  Behavioral agent (RanSMAP) : valid standalone eval")
    print("  Fusion pipeline            : semantic-driven; behavioral is cross-dataset noise")
    print("  For true fusion eval       : provide a dataset with matched semantic+behavioral per sample")
    print("="*60)
