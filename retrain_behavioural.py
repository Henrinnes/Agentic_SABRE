"""Retrain behavioural CNN on full RanSMAP dataset (all hardware configs)."""
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PARQUET = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
CKPT_DIR = Path("checkpoints")
SEED = 42
EPOCHS = 10
BATCH = 256
LR = 1e-3
DROPOUT = 0.3

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        linear_in = ((input_dim + 1) // 2) * 16 if input_dim % 2 else (input_dim // 2) * 16
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(linear_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows | configs: {df.groupby(['partition','cpu','mem']).ngroups}")
    print(f"  Ransomware: {(df['label']==1).sum():,}  Benign: {(df['label']==0).sum():,}")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print("Applying SMOTE ...")
    sm = SMOTE(random_state=SEED)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {int((y_train_r==1).sum()):,} ransomware, {int((y_train_r==0).sum()):,} benign")

    train_ds = TensorDataset(
        torch.tensor(X_train_r).unsqueeze(1),
        torch.tensor(y_train_r),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).unsqueeze(1),
        torch.tensor(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH)

    model = CNNBehavioralAgent(len(FEATURE_COLS), DROPOUT).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCELoss()

    print(f"Training for {EPOCHS} epochs ...")
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
        print(f"  Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_ds):.4f}")

    model.eval()
    preds, probs, targs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(device)).cpu()
            probs.extend(p.numpy())
            preds.extend((p > 0.5).int().numpy())
            targs.extend(yb.int().numpy())

    acc = accuracy_score(targs, preds)
    auc = roc_auc_score(targs, probs)
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test AUC-ROC  : {auc:.4f}")
    print(classification_report(targs, preds, target_names=["benign", "ransomware"]))

    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "behavioral_cnn.pt")
    joblib.dump(scaler, CKPT_DIR / "behavioral_scaler.joblib")

    meta_path = CKPT_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["behavioral_input_dim"] = len(FEATURE_COLS)
    meta["behavioral_training_data"] = PARQUET
    meta["behavioral_configs"] = int(df.groupby(["partition", "cpu", "mem"]).ngroups)
    meta["behavioral_train_rows"] = int(len(df))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved → {CKPT_DIR}/behavioral_cnn.pt, behavioral_scaler.joblib, metadata.json")


if __name__ == "__main__":
    main()
