#!/usr/bin/env python
"""
train_ensemble_agents.py
========================
Train M independent CNN members per agent for Deep Ensemble UQ comparison.

Each member is initialised with a different random seed, trained on the
same dataset split (same stratified train/test partition, seed-independent),
but with a different weight initialisation and mini-batch ordering.

Output layout
-------------
checkpoints/ensemble/
  semantic/
    seed_0.pt … seed_{M-1}.pt
  behavioral/
    seed_0.pt … seed_{M-1}.pt
  knowledge/
    seed_0.pt … seed_{M-1}.pt
    scaler.joblib          ← shared StandardScaler (fitted once)
  ensemble_results.json    ← per-member accuracy + AUC

Usage
-----
  # Train all three agents, 5 members each (default)
  python train_ensemble_agents.py

  # Train only semantic, 7 members
  python train_ensemble_agents.py --agents semantic --n_members 7

  # Dry-run: print config without training
  python train_ensemble_agents.py --dry_run

References
----------
  Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles", NeurIPS.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

# ── Architecture imports (reuse exact same classes) ───────────────────────────
from Agentic_SABRE.agents.semantic    import CNNSemanticAgent
from Agentic_SABRE.agents.behavioural import CNNBehavioralAgent
from Agentic_SABRE.agents.knowledge   import CNNKnowledgeAgent

# ── Default dataset paths ─────────────────────────────────────────────────────
SEM_PARQUET  = r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
BEH_PARQUET  = r"/mnt/4TBSTORE/Dr_Kabuye4TB/Batched/combined_i3-gen12_datasetF.parquet"
KNOW_CSV     = r"/home/henrykabuye/Ransap/Publications 2026/Proposal_2026/Knowledge_Agent/outputs/Knowledge_Agent_Features_307.csv"

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
    "Dominant_Tactic", "Platform_Signal", "Family_Count",
    "Ransomware_Context",
]


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Core training function ────────────────────────────────────────────────────

def train_one_member(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    """
    Train model on (X_train, y_train) and evaluate on (X_test, y_test).

    Returns (accuracy, auc).
    """
    set_seed(seed)   # controls weight init AND mini-batch ordering

    # Tensors — CNN expects [B, 1, D]
    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_te.to(device)).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds))
    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError:
        auc = float("nan")
    return acc, auc


# ── Per-agent training routines ───────────────────────────────────────────────

def train_semantic_ensemble(
    out_dir: Path,
    n_members: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> List[Dict]:
    print(f"\n{'='*55}")
    print(f"Semantic ensemble  ({n_members} members)")
    print(f"{'='*55}")

    df = pd.read_parquet(SEM_PARQUET)
    X  = np.stack(df["embedding"].values).astype(np.float32)
    y  = df["label"].astype(np.float32).values

    # Fixed split (same for all members — only init+order varies)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train {len(X_train):,}  |  Test {len(X_test):,}  |  Input dim {X.shape[1]}")

    input_dim = X.shape[1]
    results = []
    for i in range(n_members):
        set_seed(i)
        model = CNNSemanticAgent(input_dim, dropout=0.3)
        acc, auc = train_one_member(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, seed=i,
        )
        ckpt = out_dir / f"seed_{i}.pt"
        torch.save(model.state_dict(), str(ckpt))
        results.append({"agent": "semantic", "seed": i, "acc": acc, "auc": auc})
        print(f"  seed {i}  acc={acc:.4f}  auc={auc:.4f}  -> {ckpt.name}")

    return results


def train_behavioral_ensemble(
    out_dir: Path,
    n_members: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> List[Dict]:
    print(f"\n{'='*55}")
    print(f"Behavioral ensemble  ({n_members} members)")
    print(f"{'='*55}")

    df = pd.read_parquet(BEH_PARQUET)
    beh_cols = [c for c in BEHAVIOR_COLUMNS if c in df.columns]
    X = df[beh_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Apply SMOTE once — same resampled set for all members (matches MC-Dropout training)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  Train {len(X_train):,} (after SMOTE)  |  Test {len(X_test):,}  |  Input dim {X.shape[1]}")

    input_dim = X.shape[1]
    results = []
    for i in range(n_members):
        set_seed(i)
        model = CNNBehavioralAgent(input_dim, dropout=0.3)
        acc, auc = train_one_member(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, seed=i,
        )
        ckpt = out_dir / f"seed_{i}.pt"
        torch.save(model.state_dict(), str(ckpt))
        results.append({"agent": "behavioral", "seed": i, "acc": acc, "auc": auc})
        print(f"  seed {i}  acc={acc:.4f}  auc={auc:.4f}  -> {ckpt.name}")

    return results


def train_knowledge_ensemble(
    out_dir: Path,
    n_members: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> List[Dict]:
    print(f"\n{'='*55}")
    print(f"Knowledge ensemble  ({n_members} members)")
    print(f"{'='*55}")

    df = pd.read_csv(KNOW_CSV)
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) | (df["Tool_Total_Mentions"] > 2)
    ).astype(int)
    X_raw = df.drop(columns=KNOW_DROP_COLS, errors="ignore").select_dtypes(include="number")
    y = df["Ransomware_Context"].astype(np.float32).values

    # Fixed split — scaler fitted ONCE on seed-0 train split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df).astype(np.float32)
    X_test  = scaler.transform(X_test_df).astype(np.float32)

    # Save shared scaler
    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  Saved scaler -> {scaler_path.name}")
    print(f"  Train {len(X_train):,}  |  Test {len(X_test):,}  |  Input dim {X_train.shape[1]}")

    input_dim = X_train.shape[1]
    results = []
    for i in range(n_members):
        set_seed(i)
        model = CNNKnowledgeAgent(input_dim=input_dim, dropout=0.3).to(device)
        # Materialise LazyLinear
        with torch.no_grad():
            _ = model(torch.zeros(1, 1, input_dim, device=device))

        acc, auc = train_one_member(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, seed=i,
        )
        ckpt = out_dir / f"seed_{i}.pt"
        torch.save(model.state_dict(), str(ckpt))
        results.append({"agent": "knowledge", "seed": i, "acc": acc, "auc": auc})
        print(f"  seed {i}  acc={acc:.4f}  auc={auc:.4f}  -> {ckpt.name}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global SEM_PARQUET, BEH_PARQUET, KNOW_CSV
    parser = argparse.ArgumentParser(
        description="Train deep ensemble members for SABRE UQ comparison"
    )
    parser.add_argument(
        "--agents", nargs="+",
        choices=["semantic", "behavioral", "knowledge"],
        default=["semantic", "behavioral", "knowledge"],
        help="Which agents to train (default: all)",
    )
    parser.add_argument("--n_members", type=int, default=5,
                        help="Number of ensemble members per agent (default: 5)")
    parser.add_argument("--out_dir",   default="checkpoints/ensemble",
                        help="Root output directory for ensemble checkpoints")
    parser.add_argument("--sem_parquet",  default=SEM_PARQUET)
    parser.add_argument("--beh_parquet",  default=BEH_PARQUET)
    parser.add_argument("--know_csv",     default=KNOW_CSV)
    parser.add_argument("--sem_epochs",   type=int, default=15)
    parser.add_argument("--beh_epochs",   type=int, default=15)
    parser.add_argument("--know_epochs",  type=int, default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print config and exit without training")
    args = parser.parse_args()

    # Allow overriding paths via CLI
    SEM_PARQUET = args.sem_parquet
    BEH_PARQUET = args.beh_parquet
    KNOW_CSV    = args.know_csv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root   = Path(args.out_dir)

    print(f"Device       : {device}")
    print(f"n_members    : {args.n_members}")
    print(f"Agents       : {args.agents}")
    print(f"Output root  : {root}")

    if args.dry_run:
        print("\n[dry_run] Exiting without training.")
        return

    all_results = []

    if "semantic" in args.agents:
        d = root / "semantic"; d.mkdir(parents=True, exist_ok=True)
        all_results += train_semantic_ensemble(
            d, args.n_members, device, epochs=args.sem_epochs, lr=args.lr
        )

    if "behavioral" in args.agents:
        d = root / "behavioral"; d.mkdir(parents=True, exist_ok=True)
        all_results += train_behavioral_ensemble(
            d, args.n_members, device, epochs=args.beh_epochs, lr=args.lr
        )

    if "knowledge" in args.agents:
        d = root / "knowledge"; d.mkdir(parents=True, exist_ok=True)
        all_results += train_knowledge_ensemble(
            d, args.n_members, device, epochs=args.know_epochs, lr=args.lr
        )

    # Save summary
    summary_path = root / "ensemble_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary saved -> {summary_path}")

    # Pretty-print aggregated stats
    print("\n── Ensemble summary ──────────────────────────────────")
    for agent_name in args.agents:
        rows = [r for r in all_results if r["agent"] == agent_name]
        if not rows:
            continue
        accs = [r["acc"] for r in rows]
        aucs = [r["auc"] for r in rows]
        print(
            f"  {agent_name:12s}  "
            f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
            f"auc={np.mean(aucs):.4f}±{np.std(aucs):.4f}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
