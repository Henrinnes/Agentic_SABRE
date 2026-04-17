# train_cnn_agents.py  (run from repo root)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import random

# ---------- utils ----------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_tensor(x, unsqueeze=False):
    t = torch.tensor(x, dtype=torch.float32)
    return t.unsqueeze(1) if unsqueeze else t

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- load data ----------
# adjust these paths to your real parquet locations
df_sem = pd.read_parquet(r'C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet')
X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
y_sem = df_sem["label"].astype(np.float32).values

df_beh = pd.read_parquet(r"C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\combined_i3-gen12_datasetF.parquet")
behavior_columns = [
    'avg_read_throughput', 'avg_write_throughput',
    'read_LBA_variance', 'write_LBA_variance',
    'write_entropy_mean', 'mem_read_entropy_mean', 'mem_write_entropy_mean',
    'mem_readwrite_entropy_mean', 'mem_read_GPA_variance',
    'mem_write_GPA_variance', 'mem_readwrite_GPA_variance',
    'mem_exec_GPA_variance', 'mem_read_count', 'mem_write_count',
    'mem_readwrite_count', 'mem_exec_count'
]
X_beh = df_beh[behavior_columns].to_numpy(dtype=np.float32)
y_beh = df_beh["label"].astype(np.float32).values

# ---------- model defs ----------
class CNNSemanticAgent(nn.Module):
    def __init__(self, input_dim, dropout):
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
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(((input_dim + 1)//2) * 16 if input_dim % 2 else (input_dim//2)*16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

# def train_cnn(model, X, y, epochs=15, lr=1e-3, is_cnn=True):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, test_size=0.2, random_state=42
#     )
#     sm = SMOTE(random_state=42)
#     X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#     X_train_t = to_tensor(X_train_res, unsqueeze=is_cnn)
#     y_train_t = to_tensor(y_train_res)
#     X_test_t  = to_tensor(X_test, unsqueeze=is_cnn)
#     y_test_t  = to_tensor(y_test)

#     model.to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     crit = nn.BCELoss()

#     train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
#     test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),  batch_size=32)

#     for _ in range(epochs):
#         model.train()
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             p = model(xb)
#             loss = crit(p, yb)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#     model.eval()
#     preds, targs = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             p = model(xb)
#             preds.extend((p > 0.5).int().cpu().numpy())
#             targs.extend(yb.cpu().numpy())
#     acc = accuracy_score(targs, preds)
#     return model, acc




def train_cnn(
    model,
    X,
    y,
    epochs=15,          # was 10
    lr=1e-3,
    is_cnn=True,
    batch_size=256,     # was 32
    use_smote=True,     # new flag
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    X_train_t = to_tensor(X_train_res, unsqueeze=is_cnn)
    y_train_t = to_tensor(y_train_res)
    X_test_t  = to_tensor(X_test, unsqueeze=is_cnn)
    y_test_t  = to_tensor(y_test)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader  = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            loss = crit(p, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model(xb)
            preds.extend((p > 0.5).int().cpu().numpy())
            targs.extend(yb.cpu().numpy())
    acc = accuracy_score(targs, preds)
    return model, acc


# if __name__ == "__main__":
#     print("Training Semantic CNN...")
#     sem_model, acc_sem = train_cnn(CNNSemanticAgent(X_sem.shape[1], dropout=0.3),
#                                    X_sem, y_sem, is_cnn=True)
#     print(f"Semantic accuracy: {acc_sem:.4f}")

#     print("Training Behavioural CNN...")
#     beh_model, acc_beh = train_cnn(CNNBehavioralAgent(X_beh.shape[1], dropout=0.3),
#                                    X_beh, y_beh, is_cnn=True)
#     print(f"Behavioural accuracy: {acc_beh:.4f}")

#     ckpt_dir = Path("checkpoints")
#     ckpt_dir.mkdir(exist_ok=True)
#     torch.save(sem_model.state_dict(), ckpt_dir/"semantic_cnn.pt")
#     torch.save(beh_model.state_dict(), ckpt_dir/"behavioral_cnn.pt")
#     meta = {
#         "semantic_input_dim": int(X_sem.shape[1]),
#         "behavioral_input_dim": int(X_beh.shape[1])
#     }
#     (ckpt_dir/"metadata.json").write_text(json.dumps(meta, indent=2))
#     print("Saved checkpoints and metadata in ./checkpoints/")


if __name__ == "__main__":
    print("Training Semantic CNN...")
    sem_model, acc_sem = train_cnn(
        CNNSemanticAgent(X_sem.shape[1], dropout=0.3),
        X_sem, y_sem,
        epochs=15,
        batch_size=256,
        is_cnn=True,
        use_smote=True,   # SMOTE as in diagram
    )
    print(f"Semantic accuracy: {acc_sem:.4f}")

    print("Training Behavioural CNN...")
    beh_model, acc_beh = train_cnn(
        CNNBehavioralAgent(X_beh.shape[1], dropout=0.3),
        X_beh, y_beh,
        epochs=10,
        batch_size=32,
        is_cnn=True,
        use_smote=True,
    )
    print(f"Behavioural accuracy: {acc_beh:.4f}")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(sem_model.state_dict(), ckpt_dir/"semantic_cnn.pt")
    torch.save(beh_model.state_dict(), ckpt_dir/"behavioral_cnn.pt")
    meta = {
        "semantic_input_dim": int(X_sem.shape[1]),
        "behavioral_input_dim": int(X_beh.shape[1])
    }
    (ckpt_dir/"metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved checkpoints and metadata in ./checkpoints/")
