import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from pathlib import Path

# Dataset locations; override the root with SABRE_DATA_DIR. The RDset parquet is
# built by data_pipeline/rdset_semantic.py (all-MiniLM-L6-v2, Name/md5 dropped first).
_DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
SEM_PARQUET = str(_DATA_DIR / "RDset" / "semantic_dataset_with_embeddings.parquet")
from sklearn.metrics import roc_auc_score

class CNNSemanticAgent(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(((input_dim + 1)//2) * 32 if input_dim % 2 else (input_dim//2)*32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df_sem = pd.read_parquet(
        SEM_PARQUET
    )
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(int).values
    D = X_sem.shape[1]
    print("Loaded semantic embeddings:", X_sem.shape)

    model = CNNSemanticAgent(D, dropout=0.3).to(device)
    model.load_state_dict(torch.load(Path("checkpoints/semantic_cnn.pt"), map_location=device))
    model.eval()

    # ---- subset for speed ----
    n_samples = min(20000, X_sem.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X_sem.shape[0], size=n_samples, replace=False)
    X = X_sem[idx]
    y = y_sem[idx]

    def predict_proba(X_arr, batch=2048):
        probs = []
        for i in range(0, len(X_arr), batch):
            xb = torch.tensor(X_arr[i:i+batch], dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                p = model(xb).detach().cpu().numpy().reshape(-1)
            probs.append(p)
        return np.concatenate(probs)

    # ---- baseline AUC ----
    p_base = predict_proba(X)
    auc_base = roc_auc_score(y, p_base)
    print(f"Baseline AUC (subset): {auc_base:.6f}")

    # ---- grouped permutation settings ----
    group_size = 8          # try 8, 16, 32 if still saturated
    n_groups = 60           # number of random groups to evaluate
    repeats = 3             # repeat each group perm and average
    print(f"Grouped permutation: group_size={group_size}, n_groups={n_groups}, repeats={repeats}")

    results = []
    for g in range(n_groups):
        dims = rng.choice(D, size=group_size, replace=False)
        drops = []
        for r in range(repeats):
            Xp = X.copy()
            for j in dims:
                rng.shuffle(Xp[:, j])
            p_perm = predict_proba(Xp)
            auc_perm = roc_auc_score(y, p_perm)
            drops.append(auc_base - auc_perm)

        drop_mean = float(np.mean(drops))
        drop_std = float(np.std(drops))
        results.append((dims.tolist(), drop_mean, drop_std))
        if (g + 1) % 10 == 0:
            print(f"Group {g+1:3d}/{n_groups}: ΔAUC={drop_mean:.6e} ± {drop_std:.2e}")

    # sort groups by mean drop
    results.sort(key=lambda x: x[1], reverse=True)

    # save to json
    import json
    out = {
        "baseline_auc": float(auc_base),
        "group_size": group_size,
        "n_groups": n_groups,
        "repeats": repeats,
        "top_groups": [
            {"dims": dims, "delta_auc_mean": dm, "delta_auc_std": ds}
            for (dims, dm, ds) in results[:20]
        ],
    }
    Path("checkpoints").mkdir(exist_ok=True)
    Path("checkpoints/semantic_perm_grouped_auc.json").write_text(json.dumps(out, indent=2))
    print("\nSaved grouped permutation results to checkpoints/semantic_perm_grouped_auc.json")

    print("\nTop-10 groups by ΔAUC:")
    for i, (dims, dm, ds) in enumerate(results[:10], 1):
        print(f"{i:2d}. ΔAUC={dm:.6e} ± {ds:.2e} | dims={dims}")

if __name__ == "__main__":
    main()
