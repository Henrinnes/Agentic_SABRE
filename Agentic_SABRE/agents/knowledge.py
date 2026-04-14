# # Agentic_SABRE/agents/knowledge.py
# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Any, Tuple, List

# import numpy as np
# import torch
# import torch.nn as nn

# from ..contracts import AgentOutput


# class CNNKnowledgeAgent(nn.Module):
#     def __init__(self, input_dim: int, dropout: float = 0.3):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout),
#             nn.Flatten(),
#         )

#         self.fc = nn.Sequential(
#             nn.LazyLinear(64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.fc(x)
#         return x.squeeze(-1)


# def _enable_dropout_only(m: nn.Module):
#     if isinstance(m, nn.Dropout):
#         m.train()


# class KnowledgeAgent:
#     """
#     Loads trained CNNKnowledgeAgent and runs MC-Dropout inference.

#     Input:
#         x = {"features": 1D numpy array}
#     """
#     name = "knowledge"

#     def __init__(
#         self,
#         checkpoint_path: str = "checkpoints/knowledge_cnn.pt",
#         metadata_path: str = "checkpoints/metadata.json",
#         dropout: float = 0.3,
#         T: int = 30,
#         device: str | None = None,
#     ):
#         self.T = T
#         self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

#         meta = json.loads(Path(metadata_path).read_text())
#         self.input_dim = int(meta["knowledge_input_dim"])

#         self.model = CNNKnowledgeAgent(input_dim=self.input_dim, dropout=dropout).to(self.device)

#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, self.input_dim, dtype=torch.float32).to(self.device)
#             _ = self.model(dummy)

#         state = torch.load(checkpoint_path, map_location=self.device)
#         self.model.load_state_dict(state, strict=True)
#         self.model.eval()

#     def _prepare_tensor(self, v: np.ndarray) -> torch.Tensor:
#         if v.ndim != 1:
#             raise ValueError(f"KnowledgeAgent expected 1D feature vector, got shape {v.shape}")
#         if len(v) != self.input_dim:
#             raise ValueError(f"KnowledgeAgent expected input_dim={self.input_dim}, got {len(v)}")
#         t = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,input_dim]
#         return t.to(self.device)

#     @torch.no_grad()
#     def _mc_dropout_predict(self, x_vec: np.ndarray) -> Tuple[float, float, List[float]]:
#         samples: List[float] = []
#         self.model.apply(_enable_dropout_only)
#         inp = self._prepare_tensor(x_vec)

#         for _ in range(self.T):
#             p = self.model(inp)
#             samples.append(float(p.item()))

#         return float(np.mean(samples)), float(np.std(samples)), samples

#     def score(self, x: Dict[str, Any]) -> AgentOutput:
#         vec = np.asarray(x["features"], dtype=np.float32)
#         p_mean, p_std, _ = self._mc_dropout_predict(vec)

#         extras = {
#             "note": "MC-Dropout over knowledge CNN",
#             "top_features": []
#         }
#         return AgentOutput(prob=p_mean, uncert=p_std, extras=extras)

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import torch
import torch.nn as nn

from ..contracts import AgentOutput


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
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)


def _enable_dropout_only(m: nn.Module):
    if isinstance(m, nn.Dropout):
        m.train()


class KnowledgeAgent:
    """
    Loads trained CNNKnowledgeAgent and runs MC-Dropout inference.

    Input:
        x = {"features": 1D numpy array}
    """
    name = "knowledge"

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/knowledge_cnn.pt",
        metadata_path: str = "checkpoints/metadata.json",
        scaler_path: str = "checkpoints/knowledge_scaler.joblib",
        dropout: float = 0.3,
        T: int = 30,
        device: str | None = None,
    ):
        self.T = T
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        meta = json.loads(Path(metadata_path).read_text())
        self.input_dim = int(meta["knowledge_input_dim"])

        # Load scaler used during training
        self.scaler = joblib.load(scaler_path)

        self.model = CNNKnowledgeAgent(input_dim=self.input_dim, dropout=dropout).to(self.device)

        # initialize LazyLinear
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim, dtype=torch.float32).to(self.device)
            _ = self.model(dummy)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _prepare_tensor(self, v: np.ndarray) -> torch.Tensor:
        if v.ndim != 1:
            raise ValueError(f"KnowledgeAgent expected 1D feature vector, got shape {v.shape}")
        if len(v) != self.input_dim:
            raise ValueError(f"KnowledgeAgent expected input_dim={self.input_dim}, got {len(v)}")

        # apply the SAME scaler used in training
        # v_scaled = self.scaler.transform(v.reshape(1, -1)).astype(np.float32)
        import pandas as pd
        cols = getattr(self.scaler, "feature_names_in_", None)
        if cols is not None:
            v_df = pd.DataFrame([v], columns=cols)
            v_scaled = self.scaler.transform(v_df).astype(np.float32)
        else:
            v_scaled = self.scaler.transform(v.reshape(1, -1)).astype(np.float32)

        t = torch.tensor(v_scaled, dtype=torch.float32).unsqueeze(1)  # [1,1,input_dim]
        return t.to(self.device)

    @torch.no_grad()
    def _mc_dropout_predict(self, x_vec: np.ndarray) -> Tuple[float, float, List[float]]:
        samples: List[float] = []
        self.model.apply(_enable_dropout_only)
        inp = self._prepare_tensor(x_vec)

        for _ in range(self.T):
            p = self.model(inp)
            samples.append(float(p.item()))

        return float(np.mean(samples)), float(np.std(samples)), samples

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["features"], dtype=np.float32)
        p_mean, p_std, _ = self._mc_dropout_predict(vec)

        extras = {
            "note": "MC-Dropout over knowledge CNN",
            "top_features": []
        }
        return AgentOutput(prob=p_mean, uncert=p_std, extras=extras)