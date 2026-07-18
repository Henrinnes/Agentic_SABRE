#train_knowledge_agent.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


# -----------------------------
# CNN Architecture
# -----------------------------
# class CNNKnowledgeAgent(nn.Module):
#     def __init__(self, input_dim: int, dropout: float = 0.3):
#         super().__init__()

#         conv_out = ((input_dim + 1) // 2) * 16 if input_dim % 2 else (input_dim // 2) * 16

#         self.model = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout),
#             nn.Flatten(),
#             nn.Linear(conv_out, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.model(x).squeeze(-1)

class CNNKnowledgeAgent(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(64),   # auto-infers input dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)



# -----------------------------
# Training script
# -----------------------------
if __name__ == "__main__":

    DATA_PATH = r"/home/henry_k/Datasets/RansKnow/Knowledge_Agent_Features_718.csv"

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # Build target
    # -----------------------------
    df["Ransomware_Context"] = (
        (df["Tactic_Total_Mentions"] > 10) |
        (df["Tool_Total_Mentions"] > 2)
    ).astype(int)

    # -----------------------------
    # Remove non-feature columns
    # -----------------------------
    drop_cols = [
        "Video_ID",
        "Channel_ID",
        "Channel_Name",
        "Video_Title",
        "YouTube_URL",
        "Transcript_Path",
        "Family_List",
        "Tool_List",
        "Dominant_Tactic",
        "Platform_Signal",
        "Family_Count",
        "Ransomware_Context",
        # v4-only metadata columns, not part of the 23-dim ATT&CK feature set
        # the paper describes -- excluded to keep input_dim/feature semantics
        # unchanged from the 307-row version.
        "Year",
        "DurationSeconds",
        "Transcript_Provider",
    ]

    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")
    y = df["Ransomware_Context"]

    print("Feature matrix:", X.shape)

    # -----------------------------
    # Train/val/test split (70/15/15, stratified on the binary label)
    # -----------------------------
    # A family-stratified split (as the paper's Implementation Details
    # section previously claimed) is not achievable here: Family_List is
    # null for 372/718 rows (52%), so there is no family annotation to
    # stratify against for over half the corpus. We stratify on the binary
    # Ransomware_Context label instead, which is fully populated.
    SPLIT_SEED = 42

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SPLIT_SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SPLIT_SEED,
        stratify=y_temp,
    )

    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Capture the held-out test set's original row indices (into `df`)
    # before X_test is overwritten by the scaled numpy array below, so
    # eval_knowledge.py can reconstruct the identical held-out split.
    test_row_indices = X_test.index.tolist()

    # -----------------------------
    # Scale features (fit on train only)
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # -----------------------------
    # SMOTE (train split only, post-scaling -- matches the Semantic and
    # Behavioural agents' training pipelines: balancing is applied to the
    # training set alone, never to val/test, to avoid leakage)
    # -----------------------------
    n_train_pre_smote = len(y_train)
    print(f"Before SMOTE: {int((y_train.values == 1).sum())} ransomware-context, "
          f"{int((y_train.values == 0).sum())} not")
    smote = SMOTE(random_state=SPLIT_SEED)
    X_train, y_train_arr = smote.fit_resample(X_train, y_train.values)
    y_train = pd.Series(y_train_arr)
    print(f"After SMOTE:  {int((y_train.values == 1).sum())} ransomware-context, "
          f"{int((y_train.values == 0).sum())} not")

    # CNN expects [B,1,input_dim]
    X_train = torch.tensor(X_train).unsqueeze(1)
    X_val = torch.tensor(X_val).unsqueeze(1)
    X_test = torch.tensor(X_test).unsqueeze(1)

    y_train = torch.tensor(y_train.values.astype(np.float32))
    y_val = torch.tensor(y_val.values.astype(np.float32))
    y_test = torch.tensor(y_test.values.astype(np.float32))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = CNNKnowledgeAgent(input_dim=X_train.shape[2]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    # -----------------------------
    # Training loop with validation-based early stopping
    # -----------------------------
    EPOCHS = 30
    PATIENCE = 6

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):

        model.train()
        losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_dev)
            val_loss = loss_fn(val_pred, y_val_dev).item()

        print(f"Epoch {epoch+1:02d} | train_loss={np.mean(losses):.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    # -----------------------------
    # Evaluation (held-out test set only)
    # -----------------------------
    model.eval()

    with torch.no_grad():

        probs = model(X_test.to(device)).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    print("\nClassification report (test set):")
    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)

    print("Test AUC:", auc)

    # -----------------------------
    # Save model checkpoint
    # -----------------------------
    ckpt_path = CHECKPOINT_DIR / "knowledge_cnn.pt"

    torch.save(model.state_dict(), ckpt_path)

    print("Saved model:", ckpt_path)

    scaler_path = CHECKPOINT_DIR / "knowledge_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print("Saved scaler:", scaler_path)

    # -----------------------------
    # Save test-set row indices so eval_knowledge.py scores the same
    # held-out split rather than the full (train+val+test) dataset
    # -----------------------------
    test_indices_path = CHECKPOINT_DIR / "knowledge_test_indices.json"
    test_indices_path.write_text(json.dumps(sorted(int(i) for i in test_row_indices)))
    print("Saved held-out test indices:", test_indices_path)

    # -----------------------------
    # Update metadata.json safely
    # -----------------------------
    metadata_path = CHECKPOINT_DIR / "metadata.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {}

    metadata["knowledge_input_dim"] = int(X.shape[1])
    metadata["knowledge_features"] = list(X.columns)
    metadata["knowledge_split"] = {
        "seed": SPLIT_SEED,
        "n_train_pre_smote": n_train_pre_smote,
        "n_train_post_smote": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "stratified_on": "Ransomware_Context (binary label; Family_List is "
                          "null for 52% of rows, so family-stratification "
                          "is not used)",
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Updated metadata:", metadata_path)

    print("\nTraining complete.")