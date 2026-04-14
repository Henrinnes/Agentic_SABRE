#train_knowledge_agent.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


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

    DATA_PATH = r"C:\Users\henri\SampleDatasets\Dataset_RansKnow\outputs\Knowledge_Agent_Features_307.csv"

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
        "Ransomware_Context"
    ]

    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")
    y = df["Ransomware_Context"]

    print("Feature matrix:", X.shape)

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # CNN expects [B,1,input_dim]
    X_train = torch.tensor(X_train).unsqueeze(1)
    X_test = torch.tensor(X_test).unsqueeze(1)

    y_train = torch.tensor(y_train.values.astype(np.float32))
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
    # Training loop
    # -----------------------------
    EPOCHS = 30

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

        print(f"Epoch {epoch+1:02d} | loss={np.mean(losses):.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()

    with torch.no_grad():

        probs = model(X_test.to(device)).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)

    print("AUC:", auc)

    # -----------------------------
    # Save model checkpoint
    # -----------------------------
    ckpt_path = CHECKPOINT_DIR / "knowledge_cnn.pt"

    torch.save(model.state_dict(), ckpt_path)

    print("Saved model:", ckpt_path)

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

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Updated metadata:", metadata_path)

    # -----------------------------
    # Save scaler
    # -----------------------------
    import joblib

    scaler_path = CHECKPOINT_DIR / "knowledge_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print("Saved scaler:", scaler_path)

    print("\nTraining complete.")