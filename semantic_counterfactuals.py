
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
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

#     def forward_logits(self, x):
#         # everything except the final sigmoid
#         return self.model[:-1](x).squeeze(-1)

# def logit(p: float) -> float:
#     p = float(np.clip(p, 1e-6, 1 - 1e-6))
#     return float(np.log(p) - np.log(1 - p))

# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     df_sem = pd.read_parquet(
#         r"/home/henrykabuye/Ransap/Datasets/R2RanSAP-20241205T121617Z-001/R2RanSAP/semantic_dataset_with_embeddings.parquet"
#     )
#     X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
#     y_sem = df_sem["label"].astype(int).values
#     D = X_sem.shape[1]
#     print("Loaded semantic embeddings:", X_sem.shape)

#     model = CNNSemanticAgent(D, dropout=0.3).to(device)
#     model.load_state_dict(torch.load(Path("checkpoints/semantic_cnn.pt"), map_location=device))
#     model.eval()

#     # # pick ransomware points but avoid ultra-saturated if possible
#     # idx_mal = np.where(y_sem == 1)[0]
#     rng = np.random.default_rng(0)
#     # sample_indices = rng.choice(idx_mal, size=min(5, len(idx_mal)), replace=False)


#     # pick borderline malicious: 0.6 <= p <= 0.95 (avoid saturation)
#     with torch.no_grad():
#         X_t = torch.tensor(X_sem, dtype=torch.float32, device=device).unsqueeze(1)
#         p_all = []
#         bs = 2048
#         for i in range(0, len(X_t), bs):
#             p_all.append(model(X_t[i:i+bs]).detach().cpu().numpy())
#     p_all = np.concatenate(p_all)

#     cand = np.where((y_sem == 1) & (p_all >= 0.60) & (p_all <= 0.95))[0]
#     if len(cand) < 5:
#         cand = np.where(y_sem == 1)[0]  # fallback

#     sample_indices = rng.choice(cand, size=min(5, len(cand)), replace=False)


#     # Target probability (near boundary) and corresponding logit
#     theta = 0.49
#     target_logit = logit(theta)

#     # Optim params (minimal counterfactual)
#     n_steps = 800
#     lr = 0.03
#     lam = 1e-2          # much stronger pull-back
#     max_step_norm = 2.0 # smaller steps
#     eps = 6.0           # trust-region radius: max ||Δz||2


#     print("Generating semantic counterfactuals (logit-based)...")

#     for idx in sample_indices:
#         z0 = X_sem[idx].copy()

#         with torch.no_grad():
#             p0 = model(torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)).item()
#             l0 = model.forward_logits(torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)).item()

#         print(f"\n=== Semantic sample index {idx} ===")
#         print(f"Original p_sem = {p0:.4f}  (logit={l0:.4f})")

#         # CF variable
#         z_cf = torch.tensor(z0, dtype=torch.float32, device=device, requires_grad=True)
#         opt = torch.optim.Adam([z_cf], lr=lr)

#         z0_t = torch.tensor(z0, dtype=torch.float32, device=device)


#         for step in range(n_steps):
#             opt.zero_grad()

#             logits = model.forward_logits(z_cf.unsqueeze(0).unsqueeze(0)).squeeze()

#             # hinge: penalize only if still above target_logit
#             margin = logits - target_logit
#             loss_flip = torch.relu(margin) ** 2

#             loss_dist = lam * torch.norm(z_cf - z0_t) ** 2
#             loss = loss_flip + loss_dist
#             loss.backward()

#             # gradient clipping
#             with torch.no_grad():
#                 gnorm = z_cf.grad.norm().clamp(min=1e-12)
#                 if gnorm > max_step_norm:
#                     z_cf.grad.mul_(max_step_norm / gnorm)

#             opt.step()

#             # ---- trust-region projection: ||z_cf - z0||2 <= eps ----
#             with torch.no_grad():
#                 delta = z_cf - z0_t
#                 dnorm = torch.norm(delta)
#                 if dnorm > eps:
#                     z_cf.copy_(z0_t + delta * (eps / dnorm))

#             # ---- early stop (check every step) ----
#             with torch.no_grad():
#                 p_now = torch.sigmoid(logits).item()
#             if p_now <= theta:
#                 break


       

#         with torch.no_grad():
#             p_cf = model(z_cf.detach().unsqueeze(0).unsqueeze(0)).item()
#             l_cf = model.forward_logits(z_cf.detach().unsqueeze(0).unsqueeze(0)).item()
#             z_cf_np = z_cf.detach().cpu().numpy()

#         delta = z_cf_np - z0
#         norm = float(np.linalg.norm(delta))
#         rel = float(norm / (np.linalg.norm(z0) + 1e-8))

#         print(f"Counterfactual p_sem = {p_cf:.4f}  (logit={l_cf:.4f})")
#         print(f"||Δz||2 = {norm:.6e}, relative = {rel:.6e}")

#         # top changed dims (normalised)
#         denom = np.linalg.norm(z0) + 1e-8
#         delta_sem = delta / denom
#         topk = np.argsort(np.abs(delta_sem))[::-1][:8]
#         print("Top-8 latent dimensions by normalised change:")
#         for j in topk:
#             print(f"  dim {j:3d}: δ_sem = {delta_sem[j]:+.4e}")

#     print("\nDone.")

# if __name__ == "__main__":
#     main()

# semantic_counterfactuals_to_csv.py  (same params/eps as your script + N=50 + CSV dump)
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

    def forward_logits(self, x):
        return self.model[:-1](x).squeeze(-1)

def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p) - np.log(1 - p))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- SAME PARAMETERS AS YOUR SCRIPT ---
    SEED = 0
    N_SAMPLES = 50

    theta = 0.49
    target_logit = logit(theta)

    n_steps = 800
    lr = 0.03
    lam = 1e-2
    max_step_norm = 2.0
    eps = 6.0  # <<< matches your code

    # --- Paths ---
    sem_path = Path(SEM_PARQUET)
    ckpt_path = Path("checkpoints/semantic_cnn.pt")
    out_csv = Path("checkpoints/semantic_counterfactuals_50.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df_sem = pd.read_parquet(sem_path)
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(int).values
    D = X_sem.shape[1]
    print("Loaded semantic embeddings:", X_sem.shape)

    # --- Load model ---
    model = CNNSemanticAgent(D, dropout=0.3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rng = np.random.default_rng(SEED)

    # --- SAME borderline candidate selection as your script ---
    with torch.no_grad():
        X_t = torch.tensor(X_sem, dtype=torch.float32, device=device).unsqueeze(1)
        p_all = []
        bs = 2048
        for i in range(0, len(X_t), bs):
            p_all.append(model(X_t[i:i+bs]).detach().cpu().numpy())
    p_all = np.concatenate(p_all)

    cand = np.where((y_sem == 1) & (p_all >= 0.60) & (p_all <= 0.95))[0]
    if len(cand) < N_SAMPLES:
        cand = np.where(y_sem == 1)[0]  # fallback

    sample_indices = rng.choice(cand, size=min(N_SAMPLES, len(cand)), replace=False)

    print(f"Generating semantic counterfactuals (logit-based) for N={len(sample_indices)} samples...")
    rows = []

    for k, idx in enumerate(sample_indices, start=1):
        z0 = X_sem[idx].copy()

        with torch.no_grad():
            x0 = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            p0 = float(model(x0).item())
            l0 = float(model.forward_logits(x0).item())

        # CF variable
        z_cf = torch.tensor(z0, dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([z_cf], lr=lr)
        z0_t = torch.tensor(z0, dtype=torch.float32, device=device)

        steps_used = 0
        for step in range(n_steps):
            steps_used = step + 1
            opt.zero_grad()

            logits = model.forward_logits(z_cf.unsqueeze(0).unsqueeze(0)).squeeze()

            margin = logits - target_logit
            loss_flip = torch.relu(margin) ** 2
            loss_dist = lam * torch.norm(z_cf - z0_t) ** 2
            loss = loss_flip + loss_dist
            loss.backward()

            # gradient clipping
            with torch.no_grad():
                gnorm = z_cf.grad.norm().clamp(min=1e-12)
                if gnorm > max_step_norm:
                    z_cf.grad.mul_(max_step_norm / gnorm)

            opt.step()

            # trust-region projection: ||z_cf - z0||2 <= eps
            with torch.no_grad():
                delta_vec = z_cf - z0_t
                dnorm = torch.norm(delta_vec)
                if dnorm > eps:
                    z_cf.copy_(z0_t + delta_vec * (eps / dnorm))

            # early stop
            with torch.no_grad():
                p_now = float(torch.sigmoid(logits).item())
            if p_now <= theta:
                break

        with torch.no_grad():
            xcf = z_cf.detach().unsqueeze(0).unsqueeze(0)
            p_cf = float(model(xcf).item())
            l_cf = float(model.forward_logits(xcf).item())
            z_cf_np = z_cf.detach().cpu().numpy()

        delta = z_cf_np - z0
        l2 = float(np.linalg.norm(delta))
        rel = float(l2 / (np.linalg.norm(z0) + 1e-8))

        denom = float(np.linalg.norm(z0) + 1e-8)
        delta_sem = delta / denom
        topk = np.argsort(np.abs(delta_sem))[::-1][:8]
        top_dims = topk.tolist()
        top_deltas = [float(delta_sem[j]) for j in topk]

        rows.append({
            "idx": int(idx),
            "p0": p0,
            "logit0": l0,
            "p_cf": p_cf,
            "logit_cf": l_cf,
            "l2": l2,
            "rel": rel,
            "steps": int(steps_used),
            "theta": float(theta),
            "eps": float(eps),
            "lam": float(lam),
            "top_dims": top_dims,
            "top_deltas": top_deltas,
        })

        if k % 10 == 0 or k == len(sample_indices):
            print(f"  done {k}/{len(sample_indices)} | last: p0={p0:.3f} -> p_cf={p_cf:.3f}, ||Δz||={l2:.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to: {out_csv.resolve()}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    main()
