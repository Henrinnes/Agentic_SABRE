#!/usr/bin/env python
"""
tune_agent_policies.py
======================
Optuna CLI to tune per-agent κ_local thresholds.

Usage
-----
    python tune_agent_policies.py \
        --sem_parquet   path/to/semantic_dataset.parquet \
        --beh_parquet   path/to/behavioral_dataset.parquet \
        --know_csv      path/to/knowledge_features.csv \
        --n_trials      100 \
        --out           checkpoints/agent_policies.json

The tuner optimises κ_local for each of the three agents independently
by maximising the F1 score on a held-out validation split under the
3-agent orchestrator with consultation enabled.

When --dry_run is passed, it prints the search space and exits without
running Optuna.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS

BEHAVIOR_COLUMNS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

KNOW_DROP_COLS = list(KNOWLEDGE_NON_FEATURE_COLUMNS)  # Agentic_SABRE/knowledge_features.py


def load_datasets(args) -> tuple:
    """Return (X_sem, X_beh, X_know, y) numpy arrays."""
    df_sem = pd.read_parquet(args.sem_parquet)
    X_sem = np.stack(df_sem["embedding"].values).astype("float32")
    y_sem = df_sem["label"].to_numpy() if "label" in df_sem.columns else None

    df_beh = pd.read_parquet(args.beh_parquet)
    beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df_beh.columns]
    X_beh = df_beh[beh_cols].to_numpy(dtype="float32")
    y_beh = df_beh["label"].to_numpy() if "label" in df_beh.columns else None

    df_know = pd.read_csv(args.know_csv)
    X_know = (
        df_know.drop(columns=KNOW_DROP_COLS, errors="ignore")
        .select_dtypes(include="number")
        .to_numpy(dtype="float32")
    )
    y_know = df_know["label"].to_numpy() if "label" in df_know.columns else None

    # Use semantic labels as primary ground truth
    y = y_sem if y_sem is not None else (y_beh if y_beh is not None else y_know)
    if y is None:
        raise ValueError("No 'label' column found in any dataset.")

    N = min(len(X_sem), len(X_beh), len(X_know), len(y))
    return X_sem[:N], X_beh[:N], X_know[:N], y[:N].astype(int)


def build_objective(X_sem, X_beh, X_know, y_true, orch, val_size=0.2):
    """Return an Optuna objective function."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    idx = np.arange(len(y_true))
    _, idx_val = train_test_split(idx, test_size=val_size, stratify=y_true, random_state=42)

    X_sem_v  = X_sem[idx_val]
    X_beh_v  = X_beh[idx_val]
    X_know_v = X_know[idx_val]
    y_val    = y_true[idx_val]

    x_sems  = [{"embedding": X_sem_v[i]}  for i in range(len(idx_val))]
    x_behs  = [{"features":  X_beh_v[i]}  for i in range(len(idx_val))]
    x_knows = [{"features":  X_know_v[i]} for i in range(len(idx_val))]

    def objective(trial):
        # Search space: κ_local per agent
        kappa_sem = trial.suggest_float("kappa_sem", 0.01, 0.50)
        kappa_beh = trial.suggest_float("kappa_beh", 0.01, 0.50)
        kappa_kno = trial.suggest_float("kappa_kno", 0.01, 0.50)

        orch._agent_policies = {
            "semantic":   {"tau_local": 0.80, "kappa_local": kappa_sem},
            "behavioral": {"tau_local": 0.80, "kappa_local": kappa_beh},
            "knowledge":  {"tau_local": 0.80, "kappa_local": kappa_kno},
        }

        results = orch.handle_batch(x_sems, x_behs, x_knows)
        preds = [1 if r["decision"] != "ALLOW" else 0 for r in results]
        return f1_score(y_val, preds, zero_division=0.0)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Tune per-agent κ_local via Optuna")
    parser.add_argument("--sem_parquet",   required=True, help="Semantic embeddings parquet")
    parser.add_argument("--beh_parquet",   required=True, help="Behavioural features parquet")
    parser.add_argument("--know_csv",      required=True, help="Knowledge features CSV")
    parser.add_argument("--n_trials",      type=int, default=100)
    parser.add_argument("--out",           default="checkpoints/agent_policies.json")
    parser.add_argument("--decision_artifact",
                        default="checkpoints/decision_fuser_3agent.json")
    parser.add_argument("--dry_run",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Search space:")
        print("  kappa_sem: float in [0.01, 0.50]")
        print("  kappa_beh: float in [0.01, 0.50]")
        print("  kappa_kno: float in [0.01, 0.50]")
        print("Objective: maximise F1 on held-out validation split.")
        return

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed.  Run: pip install optuna")
        return

    from Agentic_SABRE.orchestrator_3agent import Orchestrator

    print("Loading datasets…")
    X_sem, X_beh, X_know, y = load_datasets(args)
    print(f"  N={len(y):,}  (pos={y.sum():,}  neg={(1-y).sum():,})")

    orch = Orchestrator(decision_artifact=args.decision_artifact)

    objective = build_objective(X_sem, X_beh, X_know, y, orch)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest F1 = {study.best_value:.4f}")
    print(f"Best params: {best}")

    policies: Dict[str, Dict[str, float]] = {
        "semantic":   {"tau_local": 0.80, "kappa_local": best["kappa_sem"]},
        "behavioral": {"tau_local": 0.80, "kappa_local": best["kappa_beh"]},
        "knowledge":  {"tau_local": 0.80, "kappa_local": best["kappa_kno"]},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policies, indent=2))
    print(f"Saved agent policies → {out_path}")


if __name__ == "__main__":
    main()
