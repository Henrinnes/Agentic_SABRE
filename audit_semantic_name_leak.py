"""
Audit: does the Semantic CNN's RDset result depend on the `Name` column?

In RDset every ransomware row is named `VirusShare_<hash>` and no benign row
is, so `Name` alone separates the classes perfectly. The IEEE Access paper
drops `Name` and `md5` before modelling; data_pipeline/02_semantic_embedding_
extraction.ipynb drops `md5` but embeds `Name` as the first token of the text.

The pipeline now drops both (data_pipeline/rdset_semantic.py); this audit
reproduces the historical pipeline (paraphrase-MiniLM-L6-v2, Name embedded)
to quantify what the leak did. It retrains the Semantic CNN with train_cnn_agents.py's exact recipe
(80/20 stratified split, SMOTE, Adam 1e-3, batch 256, 15 epochs, dropout 0.3)
on two inputs that differ only in whether `Name` is embedded, and reports:

  held-out   the 20% test split train_cnn_agents.py holds out
  all-rows   every RDset row, as eval_semantic_agent.py scores it (includes
             the training rows, so it is an in-sample figure)

Both with MC-Dropout (T=10, as SemanticAgent) and deterministically. Seed 42
is the original recipe; further seeds show run-to-run spread. No checkpoint
is written.

Usage:
  SABRE_DATA_DIR=/path/to/datasets python audit_semantic_name_leak.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline.rdset_semantic import SEMANTIC_COLUMNS

ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
RDSET_CSV = DATA_DIR / "RDset" / "Ransomware.csv"
WITH_NAME = DATA_DIR / "RDset" / "legacy_paraphrase-MiniLM_with-Name.parquet"
NO_NAME = DATA_DIR / "RDset" / "legacy_paraphrase-MiniLM_without-Name.parquet"
RESULTS = ROOT / "results" / "audit_semantic_name_leak.json"
SEEDS = [42, 0, 1, 2, 3]
T_MC = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNSemanticAgent(nn.Module):
    """Identical to train_cnn_agents.CNNSemanticAgent / eval_sabre2.CNNSemantic."""

    def __init__(self, input_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(((input_dim + 1) // 2) * 32 if input_dim % 2 else (input_dim // 2) * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# The historical pipeline's settings (02_semantic_embedding_extraction.ipynb before
# the fix): paraphrase-MiniLM-L6-v2, md5 dropped, Name embedded first.
LEGACY_MODEL = "paraphrase-MiniLM-L6-v2"
# "ExportsNb" is the historical misspelling of RDset's ExportNb column; the old
# pipeline skipped it as missing, so it is kept misspelt here to reproduce it.
LEGACY_SEMANTIC_COLUMNS = ["Name"] + [("ExportsNb" if c == "ExportNb" else c) for c in SEMANTIC_COLUMNS]


def build_legacy_embeddings(path: Path, include_name: bool) -> None:
    df = pd.read_csv(RDSET_CSV, delimiter="|", engine="python")
    df["label"] = (df["legitimate"] == 0).astype(int)
    cols = [c for c in LEGACY_SEMANTIC_COLUMNS if c in df.columns and (include_name or c != "Name")]
    df["semantic_text"] = df[cols].astype(str).agg(" ".join, axis=1)
    model = SentenceTransformer(LEGACY_MODEL, device=str(device))
    emb = model.encode(df["semantic_text"].tolist(), batch_size=256, convert_to_numpy=True)
    df["embedding"] = emb.astype(np.float32).tolist()
    df[["label", "semantic_text", "embedding"]].to_parquet(path, index=False)
    print(f"Wrote {path} ({len(cols)} columns embedded, Name {'included' if include_name else 'excluded'})")


def train(X, y, seed):
    """train_cnn_agents.train_cnn for the Semantic agent (imbalance < 50:1, so plain SMOTE)."""
    set_seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)
    X_res, y_res = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
    model = CNNSemanticAgent(X.shape[1], dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_res).unsqueeze(1), torch.tensor(y_res)),
                        batch_size=256, shuffle=True)
    for _ in range(15):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, X_te, y_te


@torch.no_grad()
def predict(model, X, mc: bool):
    xt = torch.tensor(X).unsqueeze(1)
    if not mc:
        model.eval()
        return torch.cat([model(xt[i:i + 8192].to(device)).cpu() for i in range(0, len(xt), 8192)]).numpy()
    model.train()  # dropout active, as SemanticAgent's MC-Dropout
    runs = [torch.cat([model(xt[i:i + 8192].to(device)).cpu() for i in range(0, len(xt), 8192)])
            for _ in range(T_MC)]
    model.eval()
    return torch.stack(runs).mean(0).numpy()


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= 0.5)),
            "f1": float(f1_score(y, p >= 0.5)), "errors": int(((p >= 0.5) != y).sum())}


def main():
    for path, include_name in ((WITH_NAME, True), (NO_NAME, False)):
        if not path.exists():
            build_legacy_embeddings(path, include_name)
    results = {}
    for variant, path in (("with_name", WITH_NAME), ("without_name", NO_NAME)):
        df = pd.read_parquet(path, columns=["embedding", "label"])
        X = np.stack(df["embedding"].values).astype(np.float32)
        y = df["label"].astype(np.float32).values
        results[variant] = []
        for seed in SEEDS:
            model, X_te, y_te = train(X, y, seed)
            row = {"seed": seed,
                   "held_out_mc": metrics(y_te, predict(model, X_te, mc=True)),
                   "held_out_det": metrics(y_te, predict(model, X_te, mc=False)),
                   "all_rows_mc": metrics(y, predict(model, X, mc=True))}
            results[variant].append(row)
            print(f"{variant:13s} seed={seed:2d}  held-out MC: {row['held_out_mc']}  "
                  f"all-rows MC: AUROC={row['all_rows_mc']['auroc']:.6f} acc={row['all_rows_mc']['accuracy']:.6f}")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    for variant, rows in results.items():
        for key in ("held_out_mc", "all_rows_mc"):
            a = np.array([r[key]["auroc"] for r in rows])
            acc = np.array([r[key]["accuracy"] for r in rows])
            print(f"SUMMARY {variant:13s} {key:12s} AUROC {a.mean():.4f} ± {a.std():.4f}   "
                  f"accuracy {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
