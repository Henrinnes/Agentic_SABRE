# === Imports ===
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import random
import pynvml
import os

# === Set Seed for Reproducibility ===
def set_seed(seed=42):
    """Sets the seed for reproducibility across numpy, torch, and random."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# === GPU Setup ===
# Removed get_best_gpu_index and related logic that sets torch.cuda.set_device
# DataParallel handles distributing to all visible GPUs starting from cuda:0

use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

# Set the primary device to cuda:0 if CUDA is available, otherwise cpu
# This is the device where the model's parameters will primarily reside
# when DataParallel is used.
device = torch.device("cuda:0" if use_cuda else "cpu")

print(f"Using device: {device}")
print(f"Number of GPUs available: {num_gpus}")

def to_tensor(x, unsqueeze=False):
    """Converts numpy array to a float32 PyTorch tensor."""
    tensor_x = torch.tensor(x, dtype=torch.float32)
    return tensor_x.unsqueeze(1) if unsqueeze else tensor_x

# === Load Semantic Dataset ===
df_sem = None
try:
    df_sem = pd.read_parquet(r'C:\Users\henri\SampleDatasets\Dataset_RanSMAP_RDset\semantic_dataset_with_embeddings.parquet')
    X_sem = np.stack(df_sem["embedding"].values).astype(np.float32)
    y_sem = df_sem["label"].astype(np.float32).values
    print("Semantic Embedding Shape:", X_sem.shape)
    print("Semantic Labels Shape:", y_sem.shape)
except FileNotFoundError:
    print("Error: Semantic dataset not found. Please check the path.")
    exit() # Exit if a critical file is missing
except Exception as e:
    print(f"An error occurred loading semantic dataset: {e}")
    exit()

# === Load Behavioral Dataset ===
df_beh = None
try:
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
except FileNotFoundError:
    print("Error: Behavioral dataset not found. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred loading behavioral dataset: {e}")
    exit()

# === Define Models ===
class CNNSemanticAgent(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear((input_dim + 1) // 2 * 32 if input_dim % 2 else (input_dim // 2) * 32, 128),
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
            nn.Conv1d(1, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear((input_dim + 1) // 2 * 16 if input_dim % 2 else (input_dim // 2) * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

class DecisionAgent(nn.Module):
    def __init__(self, input_dim=1, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

# === Train and Evaluate Function ===
def train_and_evaluate_model(model, X, y, epochs=10, lr=1e-3, is_cnn=True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    X_train_tensor = to_tensor(X_train_resampled, unsqueeze=is_cnn)
    y_train_tensor = to_tensor(y_train_resampled)
    X_test_tensor = to_tensor(X_test, unsqueeze=is_cnn)
    y_test_tensor = to_tensor(y_test)

 
    if use_cuda and num_gpus > 1:
        print(f"Wrapping model {model.__class__.__name__} with DataParallel.")
        model = nn.DataParallel(model)
 
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

    print(f"Training {model.__class__.__name__}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device) # Move data to device for each batch
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model(xb)
            preds.extend((p > 0.5).int().cpu().numpy())
            targets.extend(yb.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return model, acc

# === Train Semantic and Behavioral Models ===
print("\n--- Training Semantic Agent ---")
# Model instantiated here (on CPU by default)
sem_model_instance = CNNSemanticAgent(X_sem.shape[1], dropout=0.3)
sem_model, acc_sem = train_and_evaluate_model(sem_model_instance, X_sem, y_sem, epochs=10, lr=1e-3, is_cnn=True)
print(f"Semantic Agent Accuracy: {acc_sem:.4f}")

print("\n--- Training Behavioral Agent ---")
beh_model_instance = CNNBehavioralAgent(X_beh.shape[1], dropout=0.3)
beh_model, acc_beh = train_and_evaluate_model(beh_model_instance, X_beh, y_beh, epochs=10, lr=1e-3, is_cnn=True)
print(f"Behavioral Agent Accuracy: {acc_beh:.4f}")

# === Train Decision Agent ===
print("\n--- Training Decision Agent ---")
sem_model_eval = sem_model.module if isinstance(sem_model, nn.DataParallel) else sem_model
beh_model_eval = beh_model.module if isinstance(beh_model, nn.DataParallel) else beh_model

sem_model_eval.eval()
beh_model_eval.eval()

with torch.no_grad():
    sem_preds_list = []
    sem_inference_loader = DataLoader(TensorDataset(to_tensor(X_sem, unsqueeze=True)), batch_size=512)
    for xb in sem_inference_loader:
        xb = xb[0].to(device) # Ensure input data for inference is on the correct device
        sem_preds_list.append(sem_model_eval(xb).cpu().numpy())
    sem_pred = np.concatenate(sem_preds_list).reshape(-1, 1)

    beh_preds_list = []
    beh_inference_loader = DataLoader(TensorDataset(to_tensor(X_beh, unsqueeze=True)), batch_size=512)
    for xb in beh_inference_loader:
        xb = xb[0].to(device) # Ensure input data for inference is on the correct device
        beh_preds_list.append(beh_model_eval(xb).cpu().numpy())
    beh_pred = np.concatenate(beh_preds_list).reshape(-1, 1)

# 1. Vertically stack predictions to create a single input feature set
#    This treats each semantic prediction as an independent sample, and
#    each behavioral prediction as an independent sample, for the DecisionAgent.
X_comb = np.vstack([sem_pred, beh_pred])

# 2. Vertically stack labels to match the combined predictions
#    Ensure labels are also column vectors before stacking.
y_comb = np.vstack([y_sem.reshape(-1, 1), y_beh.reshape(-1, 1)])

# 3. Adjust the DecisionAgent's input_dim
#    Since X_comb now has shape (total_samples, 1), the input dimension is 1.
dec_model_instance = DecisionAgent(input_dim=1, dropout=0.3) # <<< input_dim changed to 1

# Now, train the Decision Agent with the combined (vertically stacked) data
dec_model, acc_dec = train_and_evaluate_model(dec_model_instance, X_comb, y_comb, epochs=10, lr=1e-3, is_cnn=False)
print(f"Decision Agent Accuracy: {acc_dec:.4f}")


import os, json, torch, torch.nn as nn
from pathlib import Path

# 1) make a checkpoints folder
ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)

# 2) save model weights (state_dict). Use the *de-parallelized* modules.
torch.save(sem_model_eval.state_dict(), ckpt_dir / "semantic_cnn.pt")
torch.save(beh_model_eval.state_dict(), ckpt_dir / "behavioral_cnn.pt")

# 3) (optional) save metadata so the package knows input dims
metadata = {
    "semantic_input_dim": int(X_sem.shape[1]),
    "behavioral_input_dim": int(X_beh.shape[1])  # you had 16 behaviour features
}
(ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
