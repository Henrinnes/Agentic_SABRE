# semantic_grad_saliency.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

# # ---- Same CNN as in train_cnn_agents.py ----
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
#         # x: (B, 1, D)
#         return self.model(x).squeeze(-1)

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
            nn.Sigmoid(),   # keep this so checkpoint matches
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def forward_logits(self, x):
        # take everything except the final Sigmoid
        logits = self.model[:-1](x).squeeze(-1)
        return logits

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- load semantic dataset ---
    df_sem = pd.read_parquet(
        r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
    )
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(np.float32).values
    input_dim = X_sem.shape[1]
    print("Loaded semantic embeddings:", X_sem.shape)

    # --- load trained semantic CNN ---
    model = CNNSemanticAgent(input_dim, dropout=0.3).to(device)
    ckpt_path = Path("checkpoints/semantic_cnn.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- compute gradient saliency on a subset ---
    n_samples = min(1024, X_sem.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X_sem.shape[0], size=n_samples, replace=False)
    X_sub = X_sem[idx]

    # leaf tensor with requires_grad=True  --> shape (B, D)
    x = torch.tensor(X_sub, dtype=torch.float32, device=device, requires_grad=True)

    # pass unsqueezed view into the model  --> shape (B, 1, D)
    # p = model(x.unsqueeze(1))  # (B,)
    logits = model.forward_logits(x.unsqueeze(1))   # (B,)

    y_sub = torch.tensor(y_sem[idx], dtype=torch.float32, device=device)
    mask = (y_sub > 0.5).float()
    if mask.sum() < 1:
        mask = torch.ones_like(mask)

    # scalar = (p * mask).sum() / mask.sum()
    scalar = (logits * mask).sum() / mask.sum()

    # backprop to x (leaf), so x.grad will be filled
    scalar.backward()

    # x.grad shape: (B, D)
    grads = x.grad.detach().cpu().numpy()          # (n_samples, input_dim)
    abs_grads = np.abs(grads)
    mean_saliency = abs_grads.mean(axis=0)         # (D,)

    # report top-20 dimensions
    topk = 20
    order = np.argsort(mean_saliency)[::-1][:topk]
    print("\nTop semantic embedding dimensions by gradient saliency:")
    for rank, j in enumerate(order, 1):
        print(f"{rank:2d}. dim {j:4d}  |  saliency={mean_saliency[j]:.6e}")

    # save for further plotting
    np.save("checkpoints/semantic_grad_saliency.npy", mean_saliency)
    print("\nSaved mean saliency vector to checkpoints/semantic_grad_saliency.npy")


if __name__ == "__main__":
    main()
