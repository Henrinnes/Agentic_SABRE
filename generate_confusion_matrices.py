#!/usr/bin/env python3
"""
generate_confusion_matrices.py
==============================
Generate per-class confusion matrices for the Behavioral CNN across three
evaluation splits: random-stratified, family holdout, hardware holdout.

Outputs:
  TIFS3_Submission/Figures/fig_cm_binary.pdf   -- 1x3 normalised binary CMs
  TIFS3_Submission/Figures/fig_cm_recall.pdf   -- per-family recall heatmap
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
CKPT    = ROOT / "checkpoints"
SPLITS  = ROOT / "splits"
FIG_DIR = ROOT / "TIFS3_Submission" / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANSMAP = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

RANSOM_CLASSES = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]
BENIGN_CLASSES = ["AESCrypt", "Firefox", "Idle", "Office", "SDelete", "Zip"]
ALL_CLASSES    = RANSOM_CLASSES + BENIGN_CLASSES

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "figure.dpi":       150,
})
DPI = 300


# ── model definition (matches generate_paper_figures.py) ──────────────────────
class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim=16, dropout=0.3):
        super().__init__()
        pooled = (input_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Flatten(), nn.Linear(pooled * 16, 64),
            nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)


def _enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()


def mc_predict(model, X: np.ndarray, T: int = 10, device: str = "cpu"):
    model.apply(_enable_dropout)
    t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    passes = []
    with torch.no_grad():
        for _ in range(T):
            passes.append(model(t).cpu().numpy())
    return np.stack(passes).mean(0)


def load_behavioral(split: str, device: str = "cpu"):
    meta = json.loads((CKPT / "metadata.json").read_text())
    dim  = meta["behavioral_input_dim"]
    m    = CNNBehavioralAgent(dim)
    m.load_state_dict(torch.load(
        str(CKPT / f"behavioral_cnn_{split}.pt"), map_location="cpu"))
    m.eval().to(device)
    sc = joblib.load(str(CKPT / f"behavioral_scaler_{split}.joblib"))
    return m, sc


def load_split(split: str, subset: str = "test"):
    sess = set((SPLITS / split / f"{subset}_sessions.txt").read_text().splitlines())
    df   = pd.read_parquet(RANSMAP)
    sub  = df[df["session"].isin(sess)].reset_index(drop=True)
    X    = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = sub["label"].astype(int).values
    fam  = sub["class_name"].values
    return X, y, fam


# ── Figure 1: binary confusion matrices (1 × 3) ───────────────────────────────
def fig_cm_binary():
    splits = [
        ("random_stratified", "Random-Stratified"),
        ("family_holdout",    "Family Holdout"),
        ("hardware_holdout",  "Hardware Holdout"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    cmap = plt.cm.Blues

    for ax, (split, title) in zip(axes, splits):
        model, scaler = load_behavioral(split)
        X, y, _ = load_split(split, "test")
        Xs = scaler.transform(X).astype(np.float32)
        probs = mc_predict(model, Xs, T=10)
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y, preds, labels=[0, 1])
        # Normalise row-wise (recall per class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap=cmap, aspect="auto")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = cm_norm[i, j]
                raw = cm[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}\n({raw:,})",
                        ha="center", va="center", fontsize=7.5, color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Ransomware"])
        ax.set_yticklabels(["Benign", "Ransomware"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Behavioural CNN\n{title}", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

        n = len(y)
        acc = (preds == y).mean()
        ax.set_xlabel(f"Predicted label\n(n={n:,}, acc={acc:.3f})")

    plt.tight_layout(pad=0.8)
    out = FIG_DIR / "fig_cm_binary.pdf"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig_cm_binary.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 2: per-family recall heatmap ───────────────────────────────────────
def fig_cm_recall():
    splits = [
        ("random_stratified", "Random\nStratified"),
        ("family_holdout",    "Family\nHoldout"),
        ("hardware_holdout",  "Hardware\nHoldout"),
    ]

    # recall[family][split_idx]
    recall_matrix = np.zeros((len(ALL_CLASSES), len(splits)))

    for s_idx, (split, _) in enumerate(splits):
        model, scaler = load_behavioral(split)
        X, y, fam = load_split(split, "test")
        Xs = scaler.transform(X).astype(np.float32)
        probs = mc_predict(model, Xs, T=10)
        preds = (probs >= 0.5).astype(int)

        for f_idx, family in enumerate(ALL_CLASSES):
            mask = (fam == family)
            if mask.sum() == 0:
                recall_matrix[f_idx, s_idx] = float("nan")
            else:
                recall_matrix[f_idx, s_idx] = (preds[mask] == y[mask]).mean()

    # Custom diverging colormap: red (low) → white → blue (high)
    cmap = LinearSegmentedColormap.from_list(
        "recall", ["#d62728", "#ffffff", "#1f77b4"], N=256)

    fig, ax = plt.subplots(figsize=(5.0, 5.5))
    im = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    # Annotate
    for i in range(len(ALL_CLASSES)):
        for j in range(len(splits)):
            val = recall_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
            else:
                color = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=8, color=color)

    # Labels
    split_labels = [s[1] for s in splits]
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(split_labels)
    ax.set_yticks(range(len(ALL_CLASSES)))
    ax.set_yticklabels(ALL_CLASSES)
    ax.set_xlabel("Evaluation split")
    ax.set_title("Per-class accuracy by evaluation split\n(Behavioural CNN, MC Dropout T=10)",
                 fontweight="bold")

    # Separator line between ransomware and benign
    ax.axhline(len(RANSOM_CLASSES) - 0.5, color="black", lw=1.5, ls="--")
    ax.text(-0.55, (len(RANSOM_CLASSES) - 1) / 2, "Ransomware",
            ha="center", va="center", fontsize=7, rotation=90,
            color="#d62728", fontweight="bold")
    ax.text(-0.55, len(RANSOM_CLASSES) + (len(BENIGN_CLASSES) - 1) / 2, "Benign",
            ha="center", va="center", fontsize=7, rotation=90,
            color="#1f77b4", fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    plt.tight_layout(pad=0.8)

    out = FIG_DIR / "fig_cm_recall.pdf"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig_cm_recall.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating confusion matrix figures …")
    fig_cm_binary()
    fig_cm_recall()
    print("Done.")
