# === Imports ===
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import optuna

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=device)

# === Load Semantic Dataset ===
df_sem = pd.read_parquet(r'C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet')
X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
y_sem = df_sem["label"].astype(np.float32).values
print("Semantic Embedding Shape:", X_sem.shape)
print("Semantic Labels Shape:", y_sem.shape)

# === Load Behavioral Dataset ===
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
print("Behavioral Features Shape:", X_beh.shape)
print("Behavioral Labels Shape:", y_beh.shape)

# === Define Models ===
class CNNSemanticAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear((input_dim // 2) * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x).squeeze()

class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear((input_dim // 2) * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x).squeeze()

class DecisionAgent(nn.Module):
    def __init__(self, sem_dim, beh_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(sem_dim + beh_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, s, b): return self.model(torch.cat([s, b], dim=1)).squeeze()

# === Train Semantic Agent ===
def train_semantic_agent(X, y, input_dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = CNNSemanticAgent(input_dim, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(to_tensor(X_train).unsqueeze(1), to_tensor(y_train)), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(to_tensor(X_test).unsqueeze(1), to_tensor(y_test)), batch_size=256)

    model.train()
    for _ in range(15):
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb)
            preds.extend((p > 0.5).int().cpu().numpy())
            targets.extend(yb.cpu().numpy())
    print("Semantic Agent Accuracy:", accuracy_score(targets, preds))
    return model

# === Train Behavioral Agent ===
def train_behavioral_agent(X, y, input_dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = CNNBehavioralAgent(input_dim, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(to_tensor(X_train).unsqueeze(1), to_tensor(y_train)), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(to_tensor(X_test).unsqueeze(1), to_tensor(y_test)), batch_size=256)

    model.train()
    for _ in range(15):
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb)
            preds.extend((p > 0.5).int().cpu().numpy())
            targets.extend(yb.cpu().numpy())
    print("Behavioral Agent Accuracy:", accuracy_score(targets, preds))
    return model

# === Train Both Agents ===
sem_model = train_semantic_agent(X_sem, y_sem, X_sem.shape[1])
beh_model = train_behavioral_agent(X_beh, y_beh, X_beh.shape[1])