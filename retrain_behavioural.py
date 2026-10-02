"""
Retrain behavioural CNN using frozen TIFS experiment splits.

Usage
-----
    # Default — random_stratified split (canonical for paper):
    python retrain_behavioural.py

    # Other splits:
    python retrain_behavioural.py --split family_holdout
    python retrain_behavioural.py --split hardware_holdout
    python retrain_behavioural.py --split temporal_holdout

Outputs (per split)
-------------------
    checkpoints/behavioral_cnn_<split>.pt
    checkpoints/behavioral_scaler_<split>.joblib
    checkpoints/behavioral_train_log_<split>.json
    checkpoints/metadata.json  (updated)

The canonical checkpoint for the paper is behavioral_cnn_random_stratified.pt,
which is also symlinked / copied to behavioral_cnn.pt for backward compatibility.
"""
import os
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PARQUET   = str(Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets")) / "ransmap_extracted" / "ransmap_train.parquet")
SPLITS_DIR = Path("splits")
CKPT_DIR   = Path("checkpoints")

SEED    = 42
EPOCHS  = 30
BATCH   = 256
LR      = 1e-3
DROPOUT = 0.3
PATIENCE = 7        # early-stopping patience on val AUC

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        pooled = (input_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(pooled * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def load_sessions(split_name: str, subset: str) -> list[str]:
    path = SPLITS_DIR / split_name / f"{subset}_sessions.txt"
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """Returns (loss, accuracy, auc)."""
    model.eval()
    crit = nn.BCELoss()
    all_probs, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            total_loss += crit(p, yb).item() * len(xb)
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    n = len(all_labels)
    loss = total_loss / n
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = float(accuracy_score(all_labels, preds))
    auc = float(roc_auc_score(all_labels, all_probs))
    return loss, acc, auc


def main(split_name: str) -> None:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Split  : {split_name}")
    print(f"Device : {device}")

    # ------------------------------------------------------------------ #
    # Load data and apply frozen split
    # ------------------------------------------------------------------ #
    print(f"\nLoading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)

    train_sess = set(load_sessions(split_name, "train"))
    val_sess   = set(load_sessions(split_name, "val"))
    test_sess  = set(load_sessions(split_name, "test"))

    df_train = df[df["session"].isin(train_sess)].reset_index(drop=True)
    df_val   = df[df["session"].isin(val_sess)].reset_index(drop=True)
    df_test  = df[df["session"].isin(test_sess)].reset_index(drop=True)

    print(f"  Train : {len(df_train):,} windows "
          f"(ransom={int((df_train.label==1).sum()):,}  benign={int((df_train.label==0).sum()):,})")
    print(f"  Val   : {len(df_val):,} windows "
          f"(ransom={int((df_val.label==1).sum()):,}  benign={int((df_val.label==0).sum()):,})")
    print(f"  Test  : {len(df_test):,} windows "
          f"(ransom={int((df_test.label==1).sum()):,}  benign={int((df_test.label==0).sum()):,})")

    X_train = df_train[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = df_train["label"].astype(np.float32).values
    X_val   = df_val[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val   = df_val["label"].astype(np.float32).values
    X_test  = df_test[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test  = df_test["label"].astype(np.float32).values

    # ------------------------------------------------------------------ #
    # Scale (fit on train only)
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32) if len(X_test) else X_test

    print(f"\n  Scaler fitted on train: mean[0]={scaler.mean_[0]:.4e}  scale[0]={scaler.scale_[0]:.4e}")

    # ------------------------------------------------------------------ #
    # SMOTE on train only
    # ------------------------------------------------------------------ #
    print("Applying SMOTE on train ...")
    sm = SMOTE(random_state=SEED)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {int((y_train_r==1).sum()):,} ransomware, "
          f"{int((y_train_r==0).sum()):,} benign")

    train_ds = TensorDataset(
        torch.tensor(X_train_r).unsqueeze(1),
        torch.tensor(y_train_r),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val).unsqueeze(1),
        torch.tensor(y_val),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).unsqueeze(1),
        torch.tensor(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ #
    # Training with early stopping on val AUC
    # ------------------------------------------------------------------ #
    model = CNNBehavioralAgent(len(FEATURE_COLS), DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params:,}")
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCELoss()

    best_val_auc   = -1.0
    patience_count = 0
    best_state     = None
    train_log      = []

    print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE}) ...")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>8}  {'ValAcc':>7}  {'ValAUC':>7}")
    print("  " + "-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_ds)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {val_auc:>7.4f}")
        train_log.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    # ------------------------------------------------------------------ #
    # Final evaluation with best weights
    # ------------------------------------------------------------------ #
    model.load_state_dict(best_state)
    _, val_acc_best, val_auc_best = evaluate(model, val_loader, device)

    has_test = len(df_test) > 0
    if has_test:
        test_loss, test_acc, test_auc = evaluate(model, test_loader, device)

        # Full classification report on test
        model.eval()
        test_probs, test_preds, test_labels = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = model(xb.to(device)).cpu().numpy()
                test_probs.extend(p)
                test_preds.extend((p >= 0.5).astype(int))
                test_labels.extend(yb.int().numpy())

        print(f"\n{'='*55}")
        print(f"  Final results — {split_name}")
        print(f"{'='*55}")
        print(f"  Best val AUC   : {val_auc_best:.4f}")
        print(f"  Test AUC-ROC   : {test_auc:.4f}")
        print(f"  Test Accuracy  : {test_acc:.4f}")
        print(classification_report(
            test_labels, test_preds,
            target_names=["benign", "ransomware"], digits=4,
        ))
    else:
        test_acc = test_auc = None
        print(f"\n{'='*55}")
        print(f"  Final results — {split_name}  (no held-out test set for this split)")
        print(f"{'='*55}")
        print(f"  Best val AUC   : {val_auc_best:.4f}")

    # ------------------------------------------------------------------ #
    # Save checkpoints
    # ------------------------------------------------------------------ #
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_pt    = CKPT_DIR / f"behavioral_cnn_{split_name}.pt"
    ckpt_scl   = CKPT_DIR / f"behavioral_scaler_{split_name}.joblib"
    ckpt_log   = CKPT_DIR / f"behavioral_train_log_{split_name}.json"

    torch.save(model.state_dict(), ckpt_pt)
    joblib.dump(scaler, ckpt_scl)
    ckpt_log.write_text(json.dumps(train_log, indent=2))

    # Keep backward-compatible names for the canonical split
    if split_name == "random_stratified":
        torch.save(model.state_dict(), CKPT_DIR / "behavioral_cnn.pt")
        joblib.dump(scaler, CKPT_DIR / "behavioral_scaler.joblib")
        print(f"\n  Also saved as behavioral_cnn.pt / behavioral_scaler.joblib (canonical)")

    meta_path = CKPT_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[f"behavioral_{split_name}"] = {
        "split":              split_name,
        "split_seed":         SEED,
        "training_parquet":   PARQUET,
        "n_train_windows":    int(len(df_train)),
        "n_val_windows":      int(len(df_val)),
        "n_test_windows":     int(len(df_test)),
        "n_configs":          int(df.groupby(["partition","cpu","mem"]).ngroups),
        "epochs_run":         len(train_log),
        "best_val_auc":       round(val_auc_best, 4),
        "test_auc":           round(test_auc, 4) if test_auc is not None else None,
        "test_accuracy":      round(test_acc, 4) if test_acc is not None else None,
        "features":           FEATURE_COLS,
        "checkpoint":         str(ckpt_pt),
        "scaler":             str(ckpt_scl),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Saved: {ckpt_pt}")
    print(f"  Saved: {ckpt_scl}")
    print(f"  Saved: {ckpt_log}")


def _existing_split(name: str) -> str:
    if not (SPLITS_DIR / name / "train_sessions.txt").exists():
        raise argparse.ArgumentTypeError(
            f"no splits/{name}/train_sessions.txt found")
    return name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="random_stratified", type=_existing_split,
                   help="Any split name with a splits/<name>/ directory "
                        "(train_sessions.txt, val_sessions.txt, "
                        "test_sessions.txt — test may be empty).")
    args = p.parse_args()
    main(args.split)
