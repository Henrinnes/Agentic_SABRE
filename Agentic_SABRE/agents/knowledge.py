# Agentic_SABRE/agents/knowledge.py
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
        self.T = T
        self.min_T = min_T
        self.max_T = max_T
        self.tol = tol
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        meta = json.loads(Path(metadata_path).read_text())
        self.input_dim = int(meta["knowledge_input_dim"])

        # Load scaler used during training
        self.scaler = joblib.load(scaler_path)
        self._scaler_cols = getattr(self.scaler, "feature_names_in_", None)

        self.model = CNNKnowledgeAgent(input_dim=self.input_dim, dropout=dropout).to(self.device)

        # initialize LazyLinear
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim, dtype=torch.float32).to(self.device)
            _ = self.model(dummy)

        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _scale(self, v: np.ndarray) -> np.ndarray:
        """Apply training scaler to a 1D feature vector, returns 1D float32."""
        if self._scaler_cols is not None:
            import pandas as pd
            v_df = pd.DataFrame([v], columns=self._scaler_cols)
            return self.scaler.transform(v_df).astype(np.float32).squeeze(0)
        return self.scaler.transform(v.reshape(1, -1)).astype(np.float32).squeeze(0)

    def _scale_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Apply training scaler to a 2D array [B, D], returns [B, D] float32."""
        if self._scaler_cols is not None:
            import pandas as pd
            df = pd.DataFrame(vecs, columns=self._scaler_cols)
            return self.scaler.transform(df).astype(np.float32)
        return self.scaler.transform(vecs).astype(np.float32)

    def _prepare(self, v: np.ndarray) -> torch.Tensor:
        if v.ndim != 1:
            raise ValueError(f"KnowledgeAgent expected 1D feature vector, got shape {v.shape}")
        if len(v) != self.input_dim:
            raise ValueError(f"KnowledgeAgent expected input_dim={self.input_dim}, got {len(v)}")
        scaled = self._scale(v)
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,D]
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
        extras = {"note": "knowledge MC Dropout", "passes_used": len(samples)}
        return AgentOutput(prob=mean_p, uncert=std_p, extras=extras)

    @torch.no_grad()
    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[AgentOutput]:
        """MC Dropout over a batch. Runs passes on [B,1,D] — stops early when uncertainty converges."""
        vecs = np.stack([np.asarray(x["features"], dtype=np.float32) for x in inputs])  # [B, D]
        scaled = self._scale_batch(vecs)  # [B, D]
        t = torch.from_numpy(scaled).unsqueeze(1).to(self.device)  # [B, 1, D]
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
        return [AgentOutput(prob=float(m), uncert=float(s), extras={"note": "knowledge MC Dropout", "passes_used": passes_used})
                for m, s in zip(means, stds)]
