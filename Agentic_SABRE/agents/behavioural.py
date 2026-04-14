# Agentic_SABRE/agents/behavioural.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from ..contracts import AgentOutput

# === SAME ARCH as training ===
class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(((input_dim + 1) // 2) * 16 if input_dim % 2 else (input_dim // 2) * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: [B,1,input_dim]
        return self.model(x).squeeze(-1)

def _enable_dropout_only(m: nn.Module):
    if isinstance(m, nn.Dropout):
        m.train()

class BehavioralAgent:
    """
    Loads your trained CNNBehavioralAgent and runs MC-Dropout inference.
    Input x is a dict with key 'features' -> 1D numpy array (shape [n_features]).
    """
    name = "behavioral"

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/behavioral_cnn.pt",
        metadata_path: str = "checkpoints/metadata.json",
        dropout: float = 0.3,
        T: int | None = None,
        min_T: int = 5,
        max_T: int = 50,
        tol: float = 1e-3,
        device: str | None = None,
    ):
        """
        T       : fixed number of MC Dropout passes (set to 10 to match paper).
                  If None (default), uses adaptive mode: runs passes until
                  uncertainty converges within `tol`, between `min_T` and `max_T`.
        min_T   : minimum passes before checking convergence (adaptive mode only).
        max_T   : hard cap on passes (adaptive mode only).
        tol     : convergence threshold — stop when mean |sigma_new - sigma_old| < tol.
        """
        meta = json.loads(Path(metadata_path).read_text())
        input_dim = int(meta["behavioral_input_dim"])
        self.model = CNNBehavioralAgent(input_dim, dropout)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)

        self.T = T
        self.min_T = min_T
        self.max_T = max_T
        self.tol = tol
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
        prev_std = None
        n_passes = self.T if self.T is not None else self.max_T
        for i in range(n_passes):
            samples.append(float(self.model(inp).item()))
            if self.T is None and len(samples) >= self.min_T and len(samples) % 5 == 0:
                curr_std = float(np.std(samples))
                if prev_std is not None and abs(curr_std - prev_std) < self.tol:
                    break
                prev_std = curr_std
        return float(np.mean(samples)), float(np.std(samples)), samples

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["features"], dtype=np.float32)
        mean_p, std_p, samples = self._mc_dropout(vec)
        extras = {"note": "behavioral MC Dropout", "passes_used": len(samples)}
        return AgentOutput(prob=mean_p, uncert=std_p, extras=extras)

    @torch.no_grad()
    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[AgentOutput]:
        """MC Dropout over a batch. Runs passes on [B,1,D] — stops early when uncertainty converges."""
        vecs = np.stack([np.asarray(x["features"], dtype=np.float32) for x in inputs])  # [B, D]
        t = torch.from_numpy(vecs).unsqueeze(1).to(self.device)  # [B, 1, D]
        self.model.apply(_enable_dropout_only)
        all_preds: List[torch.Tensor] = []
        prev_std: torch.Tensor | None = None
        n_passes = self.T if self.T is not None else self.max_T
        for i in range(n_passes):
            all_preds.append(self.model(t))  # [B]
            if self.T is None and len(all_preds) >= self.min_T and len(all_preds) % 5 == 0:
                curr_std = torch.stack(all_preds).std(0)  # [B]
                if prev_std is not None and (curr_std - prev_std).abs().mean().item() < self.tol:
                    break
                prev_std = curr_std
        preds = torch.stack(all_preds)  # [passes, B]
        means = preds.mean(0).cpu().numpy()
        stds  = preds.std(0).cpu().numpy()
        passes_used = len(all_preds)
        return [AgentOutput(prob=float(m), uncert=float(s), extras={"note": "behavioral MC Dropout", "passes_used": passes_used})
                for m, s in zip(means, stds)]
