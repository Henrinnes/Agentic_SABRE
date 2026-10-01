# # semantic_permutation_importance.py
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# from sklearn.metrics import accuracy_score
# from pathlib import Path

# class CNNSemanticAgent(nn.Module):
#     def __init__(self, input_dim, dropout=0.3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout),
#             nn.Flatten(),
#             nn.Linear(((input_dim + 1)//2) * 32 if input_dim % 2 else (input_dim//2)*32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.model(x).squeeze(-1)


# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     df_sem = pd.read_parquet(
#         r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
#     )
#     X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
#     y_sem = df_sem["label"].astype(int).values
#     input_dim = X_sem.shape[1]
#     print("Loaded semantic embeddings:", X_sem.shape)

#     model = CNNSemanticAgent(input_dim, dropout=0.3).to(device)
#     ckpt_path = Path("checkpoints/semantic_cnn.pt")
#     state = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # Subsample for speed
#     n_samples = min(10000, X_sem.shape[0])
#     rng = np.random.default_rng(0)
#     idx = rng.choice(X_sem.shape[0], size=n_samples, replace=False)
#     X = X_sem[idx]
#     y = y_sem[idx]

#     def predict_proba(X_arr):
#         xs = torch.tensor(X_arr, dtype=torch.float32, device=device).unsqueeze(1)
#         with torch.no_grad():
#             p = model(xs).cpu().numpy().reshape(-1)
#         return p

#     # baseline accuracy
#     p_base = predict_proba(X)
#     y_hat_base = (p_base > 0.5).astype(int)
#     acc_base = accuracy_score(y, y_hat_base)
#     print("Baseline semantic accuracy (subset):", acc_base)

#     max_dims = input_dim  # or set e.g. 64 if you want faster
#     importance = np.zeros(max_dims, dtype=np.float32)

#     for j in range(max_dims):
#         X_perm = X.copy()
#         rng.shuffle(X_perm[:, j])
#         p_perm = predict_proba(X_perm)
#         y_hat_perm = (p_perm > 0.5).astype(int)
#         acc_perm = accuracy_score(y, y_hat_perm)
#         importance[j] = acc_base - acc_perm
#         print(f"Dim {j:3d}: acc_perm={acc_perm:.4f}, I_j={importance[j]:.6f}")

#     np.save("checkpoints/semantic_permutation_importance.npy", importance)
#     print("\nSaved permutation importance to checkpoints/semantic_permutation_importance.npy")

#     # print top-20 dims
#     topk = 20
#     order = np.argsort(importance)[::-1][:topk]
#     print("\nTop semantic dims by permutation importance:")
#     for rank, j in enumerate(order, 1):
#         print(f"{rank:2d}. dim {j:4d}  |  I_j={importance[j]:.6f}")


# if __name__ == "__main__":
#     main()

# semantic_permutation_importance.py  (AUC + NLL, avoids accuracy saturation)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import os
from pathlib import Path

# Dataset locations; override the root with SABRE_DATA_DIR. The RDset parquet is
# built by data_pipeline/rdset_semantic.py (all-MiniLM-L6-v2, Name/md5 dropped first).
_DATA_DIR = Path(os.environ.get("SABRE_DATA_DIR", "/home/henry_k/Datasets"))
SEM_PARQUET = str(_DATA_DIR / "RDset" / "semantic_dataset_with_embeddings.parquet")


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
    input_dim = X_sem.shape[1]
    print("Loaded semantic embeddings:", X_sem.shape)

    model = CNNSemanticAgent(input_dim, dropout=0.3).to(device)
    state = torch.load(Path("checkpoints/semantic_cnn.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Subsample for speed ---
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

    # --- Baseline metrics ---
    p_base = predict_proba(X)
    acc_base = accuracy_score(y, (p_base > 0.5).astype(int))
    auc_base = roc_auc_score(y, p_base)
    nll_base = log_loss(y, np.clip(p_base, 1e-6, 1 - 1e-6))

    print(f"Baseline semantic metrics (subset): acc={acc_base:.6f}, auc={auc_base:.6f}, nll={nll_base:.6f}")

    # --- Permutation importance (per dim) measured by AUC drop and NLL increase ---
    auc_drop = np.zeros(input_dim, dtype=np.float32)
    nll_increase = np.zeros(input_dim, dtype=np.float32)

    for j in range(input_dim):
        X_perm = X.copy()
        rng.shuffle(X_perm[:, j])

        p_perm = predict_proba(X_perm)
        auc_perm = roc_auc_score(y, p_perm)
        nll_perm = log_loss(y, np.clip(p_perm, 1e-6, 1 - 1e-6))

        auc_drop[j] = auc_base - auc_perm
        nll_increase[j] = nll_perm - nll_base

        if (j + 1) % 25 == 0 or j < 10:
            print(f"Dim {j:3d}: auc_perm={auc_perm:.6f}, ΔAUC={auc_drop[j]:.6e}, ΔNLL={nll_increase[j]:.6e}")

    np.save("checkpoints/semantic_perm_auc_drop.npy", auc_drop)
    np.save("checkpoints/semantic_perm_nll_increase.npy", nll_increase)
    print("\nSaved:")
    print("  checkpoints/semantic_perm_auc_drop.npy")
    print("  checkpoints/semantic_perm_nll_increase.npy")

    # --- Print top-20 by AUC drop ---
    topk = 20
    order_auc = np.argsort(auc_drop)[::-1][:topk]
    print("\nTop semantic dims by permutation (AUC drop):")
    for rank, j in enumerate(order_auc, 1):
        print(f"{rank:2d}. dim {j:4d}  |  ΔAUC={auc_drop[j]:.6e}  |  ΔNLL={nll_increase[j]:.6e}")

    # --- Print top-20 by NLL increase ---
    order_nll = np.argsort(nll_increase)[::-1][:topk]
    print("\nTop semantic dims by permutation (NLL increase):")
    for rank, j in enumerate(order_nll, 1):
        print(f"{rank:2d}. dim {j:4d}  |  ΔNLL={nll_increase[j]:.6e}  |  ΔAUC={auc_drop[j]:.6e}")


if __name__ == "__main__":
    main()
