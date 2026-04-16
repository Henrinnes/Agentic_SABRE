# Agentic_SABRE/agents/ensemble.py
"""
Deep Ensemble agents for SABRE — UQ comparison against MC-Dropout.

Each class is a drop-in replacement for its MC-Dropout counterpart:
  SemanticEnsembleAgent   ↔  SemanticAgent
  BehavioralEnsembleAgent ↔  BehavioralAgent
  KnowledgeEnsembleAgent  ↔  KnowledgeAgent

The AgentOutput contract is identical:
  prob   = mean of M deterministic member predictions
  uncert = std  of M deterministic member predictions

Checkpoints must be trained via train_ensemble_agents.py and live at:
  checkpoints/ensemble/semantic/seed_{0..M-1}.pt
  checkpoints/ensemble/behavioral/seed_{0..M-1}.pt
  checkpoints/ensemble/knowledge/seed_{0..M-1}.pt
  checkpoints/ensemble/knowledge/scaler.joblib  (shared scaler)

Reference: Lakshminarayanan et al. (2017) "Simple and Scalable Predictive
           Uncertainty Estimation using Deep Ensembles", NeurIPS.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
import torch.nn as nn

from ..contracts import AgentOutput
from .semantic    import CNNSemanticAgent
from .behavioural import CNNBehavioralAgent
from .knowledge   import CNNKnowledgeAgent


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_members(
    model_cls,
    model_kwargs: dict,
    ckpt_dir: Path,
    n_members: int,
    device: torch.device,
    init_dummy: bool = False,
) -> List[nn.Module]:
    """
    Load M deterministic CNN members from seed_{0..M-1}.pt.
    init_dummy: pass a dummy forward to initialise LazyLinear layers first.
    """
    members = []
    for i in range(n_members):
        ckpt = ckpt_dir / f"seed_{i}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Ensemble checkpoint missing: {ckpt}\n"
                f"Run train_ensemble_agents.py first."
            )
        m = model_cls(**model_kwargs).to(device)
        if init_dummy:
            # Materialise LazyLinear before loading state_dict
            dummy_dim = model_kwargs.get("input_dim", 23)
            with torch.no_grad():
                _ = m(torch.zeros(1, 1, dummy_dim, device=device))
        state = torch.load(str(ckpt), map_location=device)
        m.load_state_dict(state, strict=True)
        m.eval()
        members.append(m)
    return members


@torch.no_grad()
def _ensemble_predict_single(
    members: List[nn.Module],
    inp: torch.Tensor,          # [1, 1, D]
) -> tuple[float, float, List[float]]:
    """One sample: return (mean, std, member_probs)."""
    preds = [float(m(inp).item()) for m in members]
    return float(np.mean(preds)), float(np.std(preds)), preds


@torch.no_grad()
def _ensemble_predict_batch(
    members: List[nn.Module],
    t: torch.Tensor,            # [B, 1, D]
) -> tuple[np.ndarray, np.ndarray]:
    """Batch: stack member outputs → [M, B], return (means [B], stds [B])."""
    preds = torch.stack([m(t) for m in members])         # [M, B]
    means = preds.mean(0).cpu().numpy()                   # [B]
    stds  = preds.std(0, unbiased=False).cpu().numpy()    # [B]  ddof=0, matches np.std()
    return means, stds


# ── SemanticEnsembleAgent ─────────────────────────────────────────────────────

class SemanticEnsembleAgent:
    """
    Drop-in replacement for SemanticAgent using Deep Ensembles.

    Loads M CNNSemanticAgent members, each deterministic (dropout disabled).
    prob  = mean of member sigmoid outputs
    uncert = std  of member sigmoid outputs
    """
    name = "semantic"

    def __init__(
        self,
        ensemble_dir: str = "checkpoints/ensemble/semantic",
        metadata_path: str = "checkpoints/metadata.json",
        n_members: int = 5,
        dropout: float = 0.3,
        device: str | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        input_dim = int(meta["semantic_input_dim"])
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.members = _load_members(
            CNNSemanticAgent,
            {"input_dim": input_dim, "dropout": dropout},
            Path(ensemble_dir),
            n_members,
            self.device,
        )

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["embedding"], dtype=np.float32)
        inp = torch.from_numpy(vec).unsqueeze(0).unsqueeze(0).to(self.device)
        mean_p, std_p, member_ps = _ensemble_predict_single(self.members, inp)
        return AgentOutput(
            prob=mean_p, uncert=std_p,
            extras={
                "note": "semantic deep ensemble",
                "n_members": len(self.members),
                "member_probs": member_ps,
            },
        )

    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[AgentOutput]:
        vecs = np.stack([np.asarray(x["embedding"], dtype=np.float32) for x in inputs])
        t = torch.from_numpy(vecs).unsqueeze(1).to(self.device)  # [B, 1, D]
        means, stds = _ensemble_predict_batch(self.members, t)
        return [
            AgentOutput(
                prob=float(m), uncert=float(s),
                extras={"note": "semantic deep ensemble", "n_members": len(self.members)},
            )
            for m, s in zip(means, stds)
        ]


# ── BehavioralEnsembleAgent ───────────────────────────────────────────────────

class BehavioralEnsembleAgent:
    """
    Drop-in replacement for BehavioralAgent using Deep Ensembles.
    """
    name = "behavioral"

    def __init__(
        self,
        ensemble_dir: str = "checkpoints/ensemble/behavioral",
        metadata_path: str = "checkpoints/metadata.json",
        n_members: int = 5,
        dropout: float = 0.3,
        device: str | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        input_dim = int(meta["behavioral_input_dim"])
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.members = _load_members(
            CNNBehavioralAgent,
            {"input_dim": input_dim, "dropout": dropout},
            Path(ensemble_dir),
            n_members,
            self.device,
        )

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["features"], dtype=np.float32)
        inp = torch.from_numpy(vec).unsqueeze(0).unsqueeze(0).to(self.device)
        mean_p, std_p, member_ps = _ensemble_predict_single(self.members, inp)
        return AgentOutput(
            prob=mean_p, uncert=std_p,
            extras={
                "note": "behavioral deep ensemble",
                "n_members": len(self.members),
                "member_probs": member_ps,
            },
        )

    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[AgentOutput]:
        vecs = np.stack([np.asarray(x["features"], dtype=np.float32) for x in inputs])
        t = torch.from_numpy(vecs).unsqueeze(1).to(self.device)
        means, stds = _ensemble_predict_batch(self.members, t)
        return [
            AgentOutput(
                prob=float(m), uncert=float(s),
                extras={"note": "behavioral deep ensemble", "n_members": len(self.members)},
            )
            for m, s in zip(means, stds)
        ]


# ── KnowledgeEnsembleAgent ────────────────────────────────────────────────────

class KnowledgeEnsembleAgent:
    """
    Drop-in replacement for KnowledgeAgent using Deep Ensembles.

    Shares a single scaler fitted on the training data (seed 0's split).
    """
    name = "knowledge"

    def __init__(
        self,
        ensemble_dir: str = "checkpoints/ensemble/knowledge",
        metadata_path: str = "checkpoints/metadata.json",
        n_members: int = 5,
        dropout: float = 0.3,
        device: str | None = None,
    ):
        meta = json.loads(Path(metadata_path).read_text())
        self.input_dim = int(meta["knowledge_input_dim"])
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Shared scaler (same training distribution for all members)
        scaler_path = Path(ensemble_dir) / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Ensemble scaler missing: {scaler_path}\n"
                f"Run train_ensemble_agents.py first."
            )
        self.scaler = joblib.load(str(scaler_path))
        self._scaler_cols = getattr(self.scaler, "feature_names_in_", None)

        self.members = _load_members(
            CNNKnowledgeAgent,
            {"input_dim": self.input_dim, "dropout": dropout},
            Path(ensemble_dir),
            n_members,
            self.device,
            init_dummy=True,
        )

    def _scale(self, v: np.ndarray) -> np.ndarray:
        if self._scaler_cols is not None:
            import pandas as pd
            return self.scaler.transform(
                pd.DataFrame([v], columns=self._scaler_cols)
            ).astype(np.float32).squeeze(0)
        return self.scaler.transform(v.reshape(1, -1)).astype(np.float32).squeeze(0)

    def _scale_batch(self, vecs: np.ndarray) -> np.ndarray:
        if self._scaler_cols is not None:
            import pandas as pd
            return self.scaler.transform(
                pd.DataFrame(vecs, columns=self._scaler_cols)
            ).astype(np.float32)
        return self.scaler.transform(vecs).astype(np.float32)

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = np.asarray(x["features"], dtype=np.float32)
        scaled = self._scale(vec)
        inp = torch.from_numpy(scaled).unsqueeze(0).unsqueeze(0).to(self.device)
        mean_p, std_p, member_ps = _ensemble_predict_single(self.members, inp)
        return AgentOutput(
            prob=mean_p, uncert=std_p,
            extras={
                "note": "knowledge deep ensemble",
                "n_members": len(self.members),
                "member_probs": member_ps,
            },
        )

    def score_batch(self, inputs: List[Dict[str, Any]]) -> List[AgentOutput]:
        vecs = np.stack([np.asarray(x["features"], dtype=np.float32) for x in inputs])
        scaled = self._scale_batch(vecs)
        t = torch.from_numpy(scaled).unsqueeze(1).to(self.device)
        means, stds = _ensemble_predict_batch(self.members, t)
        return [
            AgentOutput(
                prob=float(m), uncert=float(s),
                extras={"note": "knowledge deep ensemble", "n_members": len(self.members)},
            )
            for m, s in zip(means, stds)
        ]
