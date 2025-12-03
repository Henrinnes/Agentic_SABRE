# # Agentic_SABRE/agents/semantic.py
# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Any, Tuple, List

# import numpy as np
# import torch
# import torch.nn as nn

# from ..contracts import AgentOutput
# # from Agentic_SABRE.contracts import AgentOutput

# # === SAME ARCH as training ===
# class CNNSemanticAgent(nn.Module):
#     def __init__(self, input_dim: int, dropout: float):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, padding='same'),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Dropout(dropout),
#             nn.Flatten(),
#             nn.Linear(((input_dim + 1) // 2) * 32 if input_dim % 2 else (input_dim // 2) * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.model(x).squeeze(-1)

# def _enable_dropout_only(m: nn.Module):
#     # Put *only* Dropout layers in train() to make them active at inference
#     if isinstance(m, nn.Dropout):
#         m.train()

# class SemanticAgent:
#     """
#     Loads your trained CNNSemanticAgent and runs MC-Dropout inference.
#     Input x is a dict with key 'embedding' -> 1D numpy array (shape [input_dim]).
#     """
#     name = "semantic"

#     def __init__(
#         self,
#         checkpoint_path: str = "checkpoints/semantic_cnn.pt",
#         metadata_path: str = "checkpoints/metadata.json",
#         dropout: float = 0.3,
#         T: int = 30,
#         device: str | None = None,
#     ):
#         meta = json.loads(Path(metadata_path).read_text())
#         input_dim = int(meta["semantic_input_dim"])
#         self.model = CNNSemanticAgent(input_dim, dropout)
#         state = torch.load(checkpoint_path, map_location="cpu")
#         self.model.load_state_dict(state, strict=True)

#         self.T = T
#         self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
#         self.model.to(self.device)
#         self.model.eval()  # start in eval; we'll re-activate dropout layers per pass

#     def _prepare_tensor(self, v: np.ndarray) -> torch.Tensor:
#         # expects 1D np array -> (1, 1, input_dim)
#         t = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         return t.to(self.device)

#     @torch.no_grad()
#     def _mc_dropout_predict(self, x_vec: np.ndarray) -> Tuple[float, float, List[float]]:
#         samples: List[float] = []
#         # Activate only the dropout layers
#         self.model.apply(_enable_dropout_only)

#         inp = self._prepare_tensor(x_vec)
#         for _ in range(self.T):
#             p = self.model(inp)  # shape [1]
#             samples.append(float(p.item()))
#         mean = float(np.mean(samples))
#         std = float(np.std(samples))
#         return mean, std, samples

#     def score(self, x: Dict[str, Any]) -> AgentOutput:
#         vec = np.asarray(x["embedding"], dtype=np.float32)
#         p_mean, p_std, _ = self._mc_dropout_predict(vec)
#         extras = {"note": "MC-Dropout over semantic CNN", "top_features": []}  # plug SHAP later
#         return AgentOutput(prob=p_mean, uncert=p_std, extras=extras)





# Agentic_SABRE/agents/semantic.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from ..contracts import AgentOutput

class CNNSemanticAgent(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
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

def _enable_dropout_only(m: nn.Module):
    if isinstance(m, nn.Dropout):
        m.train()

class SemanticAgent:
    name = "semantic"

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/semantic_cnn.pt",
        metadata_path: str = "checkpoints/metadata.json",
        dropout: float = 0.3,
        T: int = 30,
        device: str | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        input_dim = int(meta["semantic_input_dim"])
        self.model = CNNSemanticAgent(input_dim, dropout)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)

        self.T = T
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def _prepare(self, v: np.ndarray) -> torch.Tensor:
        t = torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return t.to(self.device)

    @torch.no_grad()
    def _mc_dropout(self, v: np.ndarray) -> Tuple[float, float, List[float]]:
        self.model.apply(_enable_dropout_only)
        inp = self._prepare(v)
        samples: List[float] = []
        for _ in range(self.T):
            p = self.model(inp)
            samples.append(float(p.item()))
        return float(np.mean(samples)), float(np.std(samples)), samples

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["embedding"], dtype=np.float32)
        mean_p, std_p, _ = self._mc_dropout(vec)
        extras = {"note": "semantic MC Dropout"}
        return AgentOutput(prob=mean_p, uncert=std_p, extras=extras)
