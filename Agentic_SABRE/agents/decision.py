from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Optional: CTGAN (installed via sdv). If unavailable, set HAS_CTG=False to skip.
try:
    from sdv.tabular import CTGAN as SDV_CTGAN
    HAS_CTG = True
except Exception:
    HAS_CTG = False

from ..contracts import AgentOutput


# -------------------- MLP Fusion Head --------------------
class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------- Temperature Scaling --------------------
class TemperatureScaler(nn.Module):
    """Single-parameter post-hoc calibration. Use logits; here we invert Sigmoid safely."""
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        p = probs.clamp(eps, 1 - eps)
        logits = torch.log(p) - torch.log(1 - p)
        T = torch.exp(self.log_T)
        calibrated = torch.sigmoid(logits / T)
        return calibrated


# -------------------- Trainer w/ optional CTGAN --------------------
@dataclass
class DecisionTrainConfig:
    use_uncertainties: bool = False        # include σ_sem, σ_beh as inputs
    ctgan_samples: int = 0                 # number of synthetic rows to add to TRAIN ONLY (0 = off)
    ctgan_epochs: int = 100                # CTGAN fit epochs (if enabled)
    hidden: int = 32
    p_drop: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 40
    early_stop_patience: int = 6
    valid_size: float = 0.2
    temperature_scale: bool = True         # post-hoc calibration on validation


class DecisionTrainer:
    """
    Train a fusion MLP on small tabular inputs:
      X columns: [p_sem, p_beh] (+ optional [σ_sem, σ_beh])
      y: {0,1}

    Optionally augments TRAIN ONLY using CTGAN (never touch validation/test).
    """
    def __init__(self, cfg: DecisionTrainConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[FusionMLP] = None
        self.temp: Optional[TemperatureScaler] = None
        self.in_dim: int = 2 + (2 if cfg.use_uncertainties else 0)

    # ---- CTGAN augmentation (train split only) ----
    def _augment_with_ctgan(self, X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_CTG or self.cfg.ctgan_samples <= 0:
            return X_tr, y_tr

        # Build a small dataframe for sdv
        import pandas as pd
        cols = [f"x{i}" for i in range(self.in_dim)]
        df = pd.DataFrame(X_tr, columns=cols)
        df["label"] = y_tr.astype(int)

        model = SDV_CTGAN(epochs=self.cfg.ctgan_epochs)
        model.fit(df)

        synth = model.sample(self.cfg.ctgan_samples)
        synth = synth.replace([np.inf, -np.inf], np.nan).dropna()
        X_syn = synth[cols].to_numpy(dtype=np.float32)
        y_syn = synth["label"].to_numpy(dtype=np.float32)

        # Concatenate (train only)
        X_aug = np.concatenate([X_tr, X_syn], axis=0)
        y_aug = np.concatenate([y_tr, y_syn], axis=0)
        return X_aug, y_aug

    # ---- core training ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTrainer":
        assert X.shape[1] == self.in_dim, f"Expected in_dim={self.in_dim}, got {X.shape[1]}"
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=self.cfg.valid_size, stratify=y, random_state=42
        )

        # CTGAN augmentation ONLY on the training split
        X_tr, y_tr = self._augment_with_ctgan(X_tr, y_tr)

        # standardize features
        self.scaler = StandardScaler().fit(X_tr)
        X_tr = self.scaler.transform(X_tr).astype(np.float32)
        X_va = self.scaler.transform(X_va).astype(np.float32)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FusionMLP(self.in_dim, hidden=self.cfg.hidden, p_drop=self.cfg.p_drop).to(device)

        tr_loader = DataLoader(TensorDataset(
            torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=self.cfg.batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(
            torch.tensor(X_va), torch.tensor(y_va)), batch_size=1024)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        bce = nn.BCELoss()

        best_va = float("inf")
        patience = self.cfg.early_stop_patience
        for epoch in range(self.cfg.max_epochs):
            # train
            self.model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                p = self.model(xb)
                loss = bce(p, yb)
                opt.zero_grad(); loss.backward(); opt.step()

            # validate
            self.model.eval()
            with torch.no_grad():
                losses = []
                for xb, yb in va_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    p = self.model(xb)
                    losses.append(bce(p, yb).item())
                va_loss = float(np.mean(losses))

            if va_loss + 1e-6 < best_va:
                best_va = va_loss
                patience = self.cfg.early_stop_patience
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience -= 1
                if patience <= 0:
                    break

        # restore best
        self.model.load_state_dict(best_state)

        # # optional temperature scaling on validation set
        # if self.cfg.temperature_scale:
        #     self.temp = TemperatureScaler()
        #     self.temp.to(device)
        #     # Fit T by minimizing NLL on validation probs
        #     optT = torch.optim.LBFGS(self.temp.parameters(), lr=0.1, max_iter=50)

        #     def nll_on_va():
        #         optT.zero_grad(set_to_none=True)
        #         nlls = []
        #         self.model.eval()
        #         with torch.no_grad():
        #             for xb, yb in va_loader:
        #                 xb, yb = xb.to(device), yb.to(device)
        #                 p = self.model(xb)
        #                 p_cal = self.temp(p)
        #                 eps = 1e-6
        #                 nll = -(yb * torch.log(p_cal.clamp(eps, 1-eps)) +
        #                         (1 - yb) * torch.log(1 - p_cal.clamp(eps, 1-eps))).mean()
        #                 nlls.append(nll)
        #         loss = torch.stack(nlls).mean()
        #         loss.backward()
        #         return loss

        #     optT.step(nll_on_va)

        # return self
        if self.cfg.temperature_scale:
            self.temp = TemperatureScaler()
            self.temp.to(device)
            optT = torch.optim.LBFGS(self.temp.parameters(), lr=0.1, max_iter=50)

            def closure():
                optT.zero_grad(set_to_none=True)
                nlls = []
                self.model.eval()
                with torch.no_grad():
                    logits_list = []
                    y_list = []
                    for xb, yb in va_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        p = self.model(xb)
                        logits_list.append(p.detach())   # detach: we only optimise T
                        y_list.append(yb.detach())
                p_all = torch.cat(logits_list, dim=0)
                y_all = torch.cat(y_list, dim=0)

                p_cal = self.temp(p_all)
                eps = 1e-6
                nll = -(y_all * torch.log(p_cal.clamp(eps, 1-eps)) +
                        (1 - y_all) * torch.log(1 - p_cal.clamp(eps, 1-eps))).mean()
                nll.backward()
                return nll

            optT.step(closure)
        return self

    # serialize minimal artifacts
    def save(self, path: str):
        payload = {
            "in_dim": self.in_dim,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "model_state": {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            "has_temp": self.temp is not None,
            "temp_logT": (float(self.temp.log_T.detach().cpu()) if self.temp else 0.0),
            "cfg": self.cfg.__dict__,
        }
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(payload))

    @staticmethod
    def load(path: str) -> "DecisionTrainer":
        import json
        from pathlib import Path
        blob = json.loads(Path(path).read_text())
        cfg = DecisionTrainConfig(**blob["cfg"])
        tr = DecisionTrainer(cfg)
        tr.in_dim = int(blob["in_dim"])
        tr.scaler = StandardScaler()
        tr.scaler.mean_ = np.array(blob["scaler_mean"], dtype=np.float32)
        tr.scaler.scale_ = np.array(blob["scaler_scale"], dtype=np.float32)
        # rebuild model
        tr.model = FusionMLP(tr.in_dim, hidden=cfg.hidden, p_drop=cfg.p_drop)
        # load weights
        state = {k: torch.tensor(v, dtype=torch.float32) for k, v in blob["model_state"].items()}
        tr.model.load_state_dict(state, strict=True)
        # temp
        if blob.get("has_temp", False):
            tr.temp = TemperatureScaler()
            tr.temp.log_T.data = torch.tensor(blob.get("temp_logT", 0.0), dtype=torch.float32)
        return tr


# -------------------- Runtime Agent API --------------------
class DecisionAgent:
    """
    Runtime fusion agent.
    - Call `score({"p_sem": float, "p_beh": float, "σ_sem": float?, "σ_beh": float?})`
    - Returns AgentOutput(prob, uncert=fusion_std_proxy, extras=dict)
    You can:
      (a) train with DecisionTrainer and load its artifact, or
      (b) use simple mean fusion if no artifact provided.
    """
    name = "decision"

    def __init__(self, artifact_path: Optional[str] = None, use_uncertainties: bool = False):
        self.trainer: Optional[DecisionTrainer] = None
        self.use_uncert = use_uncertainties
        self.in_dim = 2 + (2 if use_uncertainties else 0)

        if artifact_path:
            self.trainer = DecisionTrainer.load(artifact_path)

    def _vectorize(self, x: Dict[str, Any]) -> np.ndarray:
        feats = [float(x.get("p_sem", 0.5)), float(x.get("p_beh", 0.5))]
        if self.use_uncert:
            feats += [float(x.get("σ_sem", 0.0)), float(x.get("σ_beh", 0.0))]
        return np.asarray(feats, dtype=np.float32)

    def _forward_trained(self, vec: np.ndarray) -> float:
        # apply scaler → torch → model → optional temperature
        xt = self.trainer.scaler.transform(vec[None, :]).astype(np.float32)
        with torch.no_grad():
            p = self.trainer.model(torch.from_numpy(xt))
            if self.trainer.temp is not None:
                p = self.trainer.temp(p)
        return float(p.squeeze().cpu().numpy())

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        vec = self._vectorize(x)
        if self.trainer is not None:
            prob = self._forward_trained(vec)
        else:
            # fallback: simple mean fusion
            prob = float((vec[0] + vec[1]) / 2.0)

        # simple uncertainty proxy at fusion (disagreement/variance across inputs)
        if self.use_uncert:
            u = float(np.std([vec[0], vec[1], vec[2] if self.use_uncert else 0.0, vec[3] if self.use_uncert else 0.0]))
        else:
            u = float(np.std([vec[0], vec[1]]))

        extras = {"inputs": {"p_sem": vec[0], "p_beh": vec[1]},
                  "uses_uncert": self.use_uncert,
                  "note": "CTGAN used at TRAIN time only" if self.trainer else "mean-fusion fallback"}
        return AgentOutput(prob=prob, uncert=u, extras=extras)

