#!/usr/bin/env python
"""
eval_ensemble_vs_mcdropout.py
==============================
Head-to-head comparison of MC-Dropout vs Deep Ensembles across all three
SABRE agents on their respective datasets.

Metrics computed per agent per method
--------------------------------------
  Accuracy          — threshold at 0.5
  ROC-AUC           — ranking quality
  Brier Score       — MSE(prob, label) — lower is better
  NLL               — negative log-likelihood — lower is better
  ECE               — Expected Calibration Error (M=15 bins) — lower is better
  Mean uncertainty  — average σ across test set
  Uncertainty @ correct   — mean σ where prediction is correct
  Uncertainty @ incorrect — mean σ where prediction is wrong (higher = better)

Usage
-----
  # Full comparison (real datasets required)
  python eval_ensemble_vs_mcdropout.py

  # Skip knowledge agent
  python eval_ensemble_vs_mcdropout.py --agents semantic behavioral

  # Save comparison figure
  python eval_ensemble_vs_mcdropout.py --plot --out_fig figures/uq_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Agentic_SABRE.knowledge_features import KNOWLEDGE_NON_FEATURE_COLUMNS

# ── Dataset paths ─────────────────────────────────────────────────────────────
SEM_PARQUET = r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
BEH_PARQUET = r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
KNOW_CSV    = r"/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv"

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


# ── UQ metric helpers ─────────────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    ECE: partition predictions into n_bins by confidence, measure
    |avg_confidence - avg_accuracy| weighted by bin size.
    Lower is better; perfectly calibrated model → ECE=0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Binary negative log-likelihood. Lower is better."""
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score = MSE(prob, label). Lower is better, range [0,1]."""
    return float(np.mean((probs - labels) ** 2))


def compute_metrics(
    probs: np.ndarray,
    uncerts: np.ndarray,
    labels: np.ndarray,
) -> dict:
    preds = (probs >= 0.5).astype(int)
    correct = (preds == labels.astype(int))
    return {
        "accuracy":        float(accuracy_score(labels, preds)),
        "auc":             float(roc_auc_score(labels, probs)),
        "brier":           brier(probs, labels),
        "nll":             nll(probs, labels),
        "ece":             expected_calibration_error(probs, labels),
        "mean_uncert":     float(np.mean(uncerts)),
        "uncert_correct":  float(np.mean(uncerts[correct]))  if correct.any()  else float("nan"),
        "uncert_wrong":    float(np.mean(uncerts[~correct])) if (~correct).any() else float("nan"),
    }


# ── Per-agent evaluation ──────────────────────────────────────────────────────

def eval_semantic(args) -> dict:
    print("\n── Semantic agent ─────────────────────────────────────")
    df = pd.read_parquet(args.sem_parquet)
    X  = np.stack(df["embedding"].values).astype(np.float32)
    y  = df["label"].astype(np.float32).values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    x_list = [{"embedding": X_test[i]} for i in range(len(X_test))]

    results = {}

    # MC-Dropout
    from Agentic_SABRE.agents.semantic import SemanticAgent
    mcd = SemanticAgent()
    print(f"  MC-Dropout: scoring {len(x_list):,} samples…", end=" ", flush=True)
    outs = mcd.score_batch(x_list)
    probs   = np.array([o.prob   for o in outs])
    uncerts = np.array([o.uncert for o in outs])
    results["mc_dropout"] = compute_metrics(probs, uncerts, y_test)
    print(f"AUC={results['mc_dropout']['auc']:.4f}  ECE={results['mc_dropout']['ece']:.4f}")

    # Deep Ensemble
    from Agentic_SABRE.agents.ensemble import SemanticEnsembleAgent
    try:
        ens = SemanticEnsembleAgent(n_members=args.n_members)
        print(f"  Ensemble:   scoring {len(x_list):,} samples…", end=" ", flush=True)
        outs = ens.score_batch(x_list)
        probs   = np.array([o.prob   for o in outs])
        uncerts = np.array([o.uncert for o in outs])
        results["deep_ensemble"] = compute_metrics(probs, uncerts, y_test)
        print(f"AUC={results['deep_ensemble']['auc']:.4f}  ECE={results['deep_ensemble']['ece']:.4f}")
    except FileNotFoundError as e:
        print(f"  SKIP ensemble — {e}")

    return results


def eval_behavioral(args) -> dict:
    print("\n── Behavioral agent ────────────────────────────────────")
    df = pd.read_parquet(args.beh_parquet)
    beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df.columns]
    X = df[beh_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(np.float32).values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    x_list = [{"features": X_test[i]} for i in range(len(X_test))]

    results = {}

    from Agentic_SABRE.agents.behavioural import BehavioralAgent
    mcd = BehavioralAgent()
    print(f"  MC-Dropout: scoring {len(x_list):,} samples…", end=" ", flush=True)
    outs = mcd.score_batch(x_list)
    probs   = np.array([o.prob   for o in outs])
    uncerts = np.array([o.uncert for o in outs])
    results["mc_dropout"] = compute_metrics(probs, uncerts, y_test)
    print(f"AUC={results['mc_dropout']['auc']:.4f}  ECE={results['mc_dropout']['ece']:.4f}")

    from Agentic_SABRE.agents.ensemble import BehavioralEnsembleAgent
    try:
        ens = BehavioralEnsembleAgent(n_members=args.n_members)
        print(f"  Ensemble:   scoring {len(x_list):,} samples…", end=" ", flush=True)
        outs = ens.score_batch(x_list)
        probs   = np.array([o.prob   for o in outs])
        uncerts = np.array([o.uncert for o in outs])
        results["deep_ensemble"] = compute_metrics(probs, uncerts, y_test)
        print(f"AUC={results['deep_ensemble']['auc']:.4f}  ECE={results['deep_ensemble']['ece']:.4f}")
    except FileNotFoundError as e:
        print(f"  SKIP ensemble — {e}")

    return results


def eval_knowledge(args) -> dict:
    print("\n── Knowledge agent ─────────────────────────────────────")
    df = pd.read_csv(args.know_csv)
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) | (df["Tool_Total_Mentions"] > 2)
    ).astype(int)
    X_raw = df.drop(columns=KNOW_DROP_COLS, errors="ignore").select_dtypes(include="number")
    y = df["Ransomware_Context"].astype(np.float32).values

    _, X_test_df, _, y_test = train_test_split(
        X_raw, y, test_size=0.25, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_df, _ = train_test_split(X_raw, test_size=0.25, random_state=42)
    scaler.fit(X_train_df)
    X_test = scaler.transform(X_test_df).astype(np.float32)
    x_list = [{"features": X_test[i]} for i in range(len(X_test))]

    results = {}

    from Agentic_SABRE.agents.knowledge import KnowledgeAgent
    mcd = KnowledgeAgent()
    print(f"  MC-Dropout: scoring {len(x_list):,} samples…", end=" ", flush=True)
    outs = mcd.score_batch(x_list)
    probs   = np.array([o.prob   for o in outs])
    uncerts = np.array([o.uncert for o in outs])
    results["mc_dropout"] = compute_metrics(probs, uncerts, y_test)
    print(f"AUC={results['mc_dropout']['auc']:.4f}  ECE={results['mc_dropout']['ece']:.4f}")

    from Agentic_SABRE.agents.ensemble import KnowledgeEnsembleAgent
    try:
        ens = KnowledgeEnsembleAgent(n_members=args.n_members)
        print(f"  Ensemble:   scoring {len(x_list):,} samples…", end=" ", flush=True)
        outs = ens.score_batch(x_list)
        probs   = np.array([o.prob   for o in outs])
        uncerts = np.array([o.uncert for o in outs])
        results["deep_ensemble"] = compute_metrics(probs, uncerts, y_test)
        print(f"AUC={results['deep_ensemble']['auc']:.4f}  ECE={results['deep_ensemble']['ece']:.4f}")
    except FileNotFoundError as e:
        print(f"  SKIP ensemble — {e}")

    return results


# ── Pretty-print comparison table ────────────────────────────────────────────

METRIC_LABELS = {
    "accuracy":       "Accuracy",
    "auc":            "ROC-AUC",
    "brier":          "Brier ↓",
    "nll":            "NLL ↓",
    "ece":            "ECE ↓",
    "mean_uncert":    "Mean σ",
    "uncert_correct": "σ @ correct",
    "uncert_wrong":   "σ @ wrong ↑",
}


def print_table(agent_name: str, results: dict):
    methods = list(results.keys())
    if not methods:
        return

    col_w = 14
    header = f"{'Metric':<18}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(f"\n  {agent_name.upper()} AGENT")
    print("  " + "─" * len(header))
    print("  " + header)
    print("  " + "─" * len(header))

    for key, label in METRIC_LABELS.items():
        row = f"  {label:<18}"
        for m in methods:
            val = results[m].get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        # Flag which method wins on each metric (lower-is-better: brier/nll/ece; higher: rest)
        vals = {m: results[m].get(key, float("nan")) for m in methods}
        if len(methods) == 2:
            a, b = methods
            if key in ("brier", "nll", "ece", "mean_uncert"):
                winner = a if vals[a] <= vals[b] else b
            elif key == "uncert_wrong":
                winner = a if vals[a] >= vals[b] else b
            else:
                winner = a if vals[a] >= vals[b] else b
            row += f"  ← {winner}"
        print(row)

    print("  " + "─" * len(header))


# ── Optional calibration plot ─────────────────────────────────────────────────

def plot_calibration_comparison(all_results: dict, out_fig: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    agents = list(all_results.keys())
    n = len(agents)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    colours = {"mc_dropout": "#2980b9", "deep_ensemble": "#e97132"}
    labels  = {"mc_dropout": "MC-Dropout", "deep_ensemble": "Deep Ensemble"}

    for ax, agent_name in zip(axes[0], agents):
        for method, metrics in all_results[agent_name].items():
            col = colours.get(method, "#888")
            lbl = labels.get(method, method)
            ax.bar(
                [lbl],
                [metrics["ece"]],
                color=col, alpha=0.85,
                label=f"{lbl} ECE={metrics['ece']:.4f}",
            )
        ax.set_title(f"{agent_name.capitalize()} — ECE comparison")
        ax.set_ylabel("ECE (lower is better)")
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)

    fig.tight_layout()
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_fig}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare MC-Dropout vs Deep Ensembles on SABRE agents"
    )
    parser.add_argument(
        "--agents", nargs="+",
        choices=["semantic", "behavioral", "knowledge"],
        default=["semantic", "behavioral", "knowledge"],
    )
    parser.add_argument("--n_members",  type=int, default=5)
    parser.add_argument("--sem_parquet",  default=SEM_PARQUET)
    parser.add_argument("--beh_parquet",  default=BEH_PARQUET)
    parser.add_argument("--know_csv",     default=KNOW_CSV)
    parser.add_argument("--plot",         action="store_true")
    parser.add_argument("--out_fig",      default="Figures/uq_comparison_ece.png")
    parser.add_argument("--out_json",     default="checkpoints/uq_comparison.json")
    args = parser.parse_args()

    eval_fns = {
        "semantic":   eval_semantic,
        "behavioral": eval_behavioral,
        "knowledge":  eval_knowledge,
    }

    print("=" * 60)
    print("SABRE UQ Comparison: MC-Dropout vs Deep Ensembles")
    print("=" * 60)

    all_results = {}
    for agent_name in args.agents:
        all_results[agent_name] = eval_fns[agent_name](args)

    # Print tables
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for agent_name, res in all_results.items():
        print_table(agent_name, res)

    # Save JSON
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out}")

    if args.plot:
        plot_calibration_comparison(all_results, args.out_fig)


if __name__ == "__main__":
    main()
