#!/usr/bin/env python3
"""
eval_e10_feedback_loop.py — E10 Expert Feedback Loop Simulation
================================================================
Simulates continual learning under temporal drift and OOD family arrival.

Stream design
-------------
  Phase 1 (warm-up): temporal val windows, Jun 2023 – Feb 2024 (5,494 windows)
    Model scored prequentially; no updates.
  Phase 2 (active):  temporal test + OOD variants merged by timestamp,
    Feb 2024 – Mar 2024 (5,296 + 1,397 = 6,693 windows).
    OOD Conti_01-07 variants appear naturally at Feb 26-29.
    Updates enabled. Analyst budget = BUDGET labels per step.

Update strategies
-----------------
  no_update  : score only; never retrain
  finetune   : SGD fine-tune on new labels; no forgetting protection
  replay     : fine-tune on new labels + random replay buffer from training data
  ewc        : fine-tune with EWC diagonal-Fisher penalty on all parameters

Metrics (reported per strategy)
--------------------------------
  Prequential AUPRC   — rolling 500-window AUC over entire Phase 2
  Forgetting rate     — ΔAUPRC on memory probe (original families) before vs. after update
  Recovery time       — steps until AUPRC on OOD variants ≥ 0.80
  FN during drift     — missed OOD ransomware windows before first recovery
  FP/day at drift     — false alarm rate on benign windows during OOD period
"""
from __future__ import annotations

import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANSMAP_PARQUET = "/home/henry_k/Datasets/ransmap_extracted/ransmap_train.parquet"
OOD_PARQUET     = "/home/henry_k/Datasets/ransmap_extracted/ransmap_ood_variants.parquet"
SPLITS_DIR      = Path("splits")
CKPT_DIR        = Path("checkpoints")
RESULTS_DIR     = Path("results")

SPLIT          = "temporal_holdout"
STEP_SIZE      = 200     # windows processed per time step
BUDGET         = 50      # analyst labels per step (25% labeling rate)
REPLAY_BUFFER  = 500     # samples in replay buffer
EWC_LAMBDA     = 300.0   # EWC penalty strength
LR_FINETUNE    = 5e-5    # fine-tuning learning rate
FT_EPOCHS      = 2       # epochs per update
PROBE_SIZE     = 200     # memory probe samples (forgetting measurement)
ROLLING_WIN    = 500     # rolling window for prequential AUPRC
DROPOUT        = 0.3
SEED           = 42

FEATURE_COLS = [
    "avg_read_throughput", "avg_write_throughput",
    "read_LBA_variance", "write_LBA_variance",
    "write_entropy_mean", "mem_read_entropy_mean", "mem_write_entropy_mean",
    "mem_readwrite_entropy_mean", "mem_read_GPA_variance",
    "mem_write_GPA_variance", "mem_readwrite_GPA_variance",
    "mem_exec_GPA_variance", "mem_read_count", "mem_write_count",
    "mem_readwrite_count", "mem_exec_count",
]

# ---------------------------------------------------------------------------
# Model (identical arch to training)
# ---------------------------------------------------------------------------
class CNNBehavioralAgent(nn.Module):
    def __init__(self, input_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        pooled = (input_dim + 1) // 2
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(pooled * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def load_model(device: torch.device) -> CNNBehavioralAgent:
    m = CNNBehavioralAgent()
    m.load_state_dict(torch.load(
        str(CKPT_DIR / "behavioral_cnn_temporal_holdout.pt"), map_location="cpu"))
    return m.to(device)


def score_batch(model: nn.Module, X: np.ndarray,
                device: torch.device) -> np.ndarray:
    model.eval()
    t = torch.tensor(X).unsqueeze(1).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy()


def compute_fisher(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   device: torch.device, n_samples: int = 300) -> dict[str, torch.Tensor]:
    """Diagonal Fisher information (gradient² expectation) on training samples."""
    model.train()
    crit  = nn.BCELoss()
    fisher: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p) for n, p in model.named_parameters()
    }
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    for i in idx:
        xi = torch.tensor(X[i:i+1]).unsqueeze(1).to(device)
        yi = torch.tensor(y[i:i+1], dtype=torch.float32).to(device)
        model.zero_grad()
        loss = crit(model(xi), yi)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    n = min(n_samples, len(X))
    for n_key in fisher:
        fisher[n_key] /= n
    model.eval()
    return fisher


def ewc_loss(model: nn.Module, crit: nn.Module,
             xb: torch.Tensor, yb: torch.Tensor,
             fisher: dict, theta_star: dict,
             lam: float) -> torch.Tensor:
    base = crit(model(xb), yb)
    penalty = torch.tensor(0.0, device=xb.device)
    for n, p in model.named_parameters():
        if n in fisher:
            penalty = penalty + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return base + (lam / 2) * penalty


def finetune_step(model: nn.Module, X_new: np.ndarray, y_new: np.ndarray,
                  device: torch.device,
                  X_replay: np.ndarray | None = None,
                  y_replay: np.ndarray | None = None,
                  fisher: dict | None = None,
                  theta_star: dict | None = None,
                  lam: float = EWC_LAMBDA) -> None:
    """One fine-tuning update on new labels (+ optional replay / EWC)."""
    model.train()
    opt  = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    crit = nn.BCELoss()

    if X_replay is not None and len(X_replay) > 0:
        X_all = np.concatenate([X_new, X_replay], axis=0)
        y_all = np.concatenate([y_new, y_replay], axis=0)
    else:
        X_all, y_all = X_new, y_new

    idx = np.arange(len(X_all))
    for _ in range(FT_EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, len(idx), 64):
            b    = idx[s:s+64]
            xb   = torch.tensor(X_all[b]).unsqueeze(1).to(device)
            yb   = torch.tensor(y_all[b], dtype=torch.float32).to(device)
            opt.zero_grad()
            if fisher is not None and theta_star is not None:
                loss = ewc_loss(model, crit, xb, yb, fisher, theta_star, lam)
            else:
                loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()


def rolling_auprc(labels: list[int], probs: list[float],
                  window: int = ROLLING_WIN) -> float:
    if len(labels) < 10:
        return float("nan")
    y = np.array(labels[-window:])
    p = np.array(probs[-window:])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_strategy(strategy: str,
                 phase1_X: np.ndarray, phase1_y: np.ndarray,
                 phase2_X: np.ndarray, phase2_y: np.ndarray,
                 phase2_fam: np.ndarray,
                 probe_X: np.ndarray, probe_y: np.ndarray,
                 train_X: np.ndarray, train_y: np.ndarray,
                 scaler, device: torch.device) -> dict:

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = load_model(device)

    # EWC: compute Fisher on training data, store reference weights
    fisher     = None
    theta_star = None
    if strategy == "ewc":
        print(f"    Computing Fisher information …")
        fisher = compute_fisher(model, train_X, train_y, device)
        theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Replay buffer seeded from training data
    replay_X: deque = deque(maxlen=REPLAY_BUFFER)
    replay_y: deque = deque(maxlen=REPLAY_BUFFER)
    if strategy in ("replay", "replay_ewc"):
        rng_idx = np.random.choice(len(train_X), size=min(REPLAY_BUFFER, len(train_X)),
                                   replace=False)
        for i in rng_idx:
            replay_X.append(train_X[i])
            replay_y.append(train_y[i])

    # Pending label accumulator
    pending_X: list[np.ndarray] = []
    pending_y: list[int]        = []

    # Tracking
    all_labels:     list[int]   = []
    all_probs:      list[float] = []
    roll_auprc:     list[float] = []  # one entry per step
    probe_auprc:    list[float] = []
    ood_detected:   list[int]   = []  # per step: # OOD variants correctly detected
    ood_total:      list[int]   = []  # per step: # OOD variants seen
    fn_count:       int         = 0   # false negatives on OOD during drift (before recovery)
    recovery_step:  int | None  = None
    step_families:  list[str]   = []  # dominant family per step

    n2 = len(phase2_X)
    n_steps = (n2 + STEP_SIZE - 1) // STEP_SIZE

    for step in range(n_steps):
        s = step * STEP_SIZE
        e = min(s + STEP_SIZE, n2)
        Xs = phase2_X[s:e]
        ys = phase2_y[s:e]
        fs = phase2_fam[s:e]

        # Score current batch
        probs = score_batch(model, Xs, device)
        preds = (probs >= 0.5).astype(int)

        all_labels.extend(ys.tolist())
        all_probs.extend(probs.tolist())

        # Track OOD variant detection this step
        ood_mask = np.array(["Conti_0" in f for f in fs])
        if ood_mask.any():
            ood_detected.append(int((preds[ood_mask] == 1).sum()))
            ood_total.append(int(ood_mask.sum()))
            fn_count += int(((preds[ood_mask] == 0) & (ys[ood_mask] == 1)).sum())
        else:
            ood_detected.append(0)
            ood_total.append(0)

        # Rolling AUPRC
        rauc = rolling_auprc(all_labels, all_probs)
        roll_auprc.append(rauc)

        # Probe AUC (forgetting measurement)
        p_probe = score_batch(model, probe_X, device)
        probe_auprc.append(float(average_precision_score(probe_y, p_probe))
                           if len(np.unique(probe_y)) > 1 else float("nan"))

        # Recovery: first step where OOD recall ≥ 0.80
        if recovery_step is None and ood_total[-1] > 0:
            ood_recall = ood_detected[-1] / ood_total[-1]
            if ood_recall >= 0.80:
                recovery_step = step

        # Dominant family this step
        fams, counts = np.unique(fs, return_counts=True)
        step_families.append(fams[counts.argmax()])

        # Label selection: uncertainty sampling (highest p*(1-p))
        if strategy != "no_update":
            uncert = probs * (1 - probs)
            top_k  = min(BUDGET, len(Xs))
            sel    = np.argsort(uncert)[-top_k:]
            pending_X.extend(Xs[sel])
            pending_y.extend(ys[sel].tolist())

            # Update when budget reached
            if len(pending_X) >= BUDGET:
                pX = np.stack(pending_X[:BUDGET])
                pY = np.array(pending_y[:BUDGET], dtype=np.float32)
                pending_X = pending_X[BUDGET:]
                pending_y = pending_y[BUDGET:]

                rX = (np.stack(list(replay_X)) if strategy in ("replay",) and replay_X
                      else None)
                rY = (np.array(list(replay_y), dtype=np.float32)
                      if rX is not None else None)

                finetune_step(
                    model, pX, pY, device,
                    X_replay=rX, y_replay=rY,
                    fisher=fisher if strategy == "ewc" else None,
                    theta_star=theta_star if strategy == "ewc" else None,
                )

                # Refresh replay buffer with newly labeled samples
                if strategy in ("replay",):
                    for xi, yi in zip(pX, pY):
                        replay_X.append(xi)
                        replay_y.append(int(yi))

    # Aggregate forgetting: max_probe - min_probe after first update
    probe_arr = np.array([p for p in probe_auprc if not np.isnan(p)])
    forgetting = float(probe_arr.max() - probe_arr.min()) if len(probe_arr) > 1 else 0.0

    # Overall Phase 2 AUPRC and AUROC
    y_all = np.array(all_labels)
    p_all = np.array(all_probs)
    overall_auprc = float(average_precision_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else float("nan")
    overall_auroc = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else float("nan")

    # OOD summary
    total_ood_seen = sum(ood_total)
    total_ood_det  = sum(ood_detected)
    ood_recall_overall = total_ood_det / total_ood_seen if total_ood_seen > 0 else 0.0

    return {
        "strategy":           strategy,
        "overall_auprc":      round(overall_auprc, 4),
        "overall_auroc":      round(overall_auroc, 4),
        "forgetting":         round(forgetting, 4),
        "recovery_step":      recovery_step,
        "fn_before_recovery": fn_count,
        "ood_recall_overall": round(ood_recall_overall, 4),
        "total_ood_windows":  total_ood_seen,
        "n_steps":            n_steps,
        "roll_auprc":         [round(v, 4) if not np.isnan(v) else None for v in roll_auprc],
        "probe_auprc":        [round(v, 4) if not np.isnan(v) else None for v in probe_auprc],
        "ood_detected":       ood_detected,
        "ood_total":          ood_total,
    }


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("Loading data …")
    df  = pd.read_parquet(RANSMAP_PARQUET)
    ood = pd.read_parquet(OOD_PARQUET)
    scaler = joblib.load(str(CKPT_DIR / "behavioral_scaler_temporal_holdout.joblib"))

    def get_split(subset: str) -> pd.DataFrame:
        sess = set((SPLITS_DIR / SPLIT / f"{subset}_sessions.txt").read_text().splitlines())
        return df[df["session"].isin(sess)].sort_values("window_start").reset_index(drop=True)

    df_train = get_split("train")
    df_val   = get_split("val")
    df_test  = get_split("test")

    # Merge test + OOD by timestamp (OOD appears naturally at Feb 26-29)
    ood_sorted = ood.sort_values("window_start").reset_index(drop=True)
    df_phase2  = pd.concat([df_test, ood_sorted], ignore_index=True)\
                   .sort_values("window_start").reset_index(drop=True)

    print(f"  Train  : {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}  OOD: {len(ood_sorted):,}")
    print(f"  Phase2 (test+OOD merged): {len(df_phase2):,} windows")
    print(f"    ransom={int((df_phase2.label==1).sum())}  benign={int((df_phase2.label==0).sum())}")
    print(f"    OOD variants: {sorted(df_phase2[df_phase2.class_name.str.contains('Conti_0', na=False)].class_name.unique())}")

    def to_X(df_sub: pd.DataFrame) -> np.ndarray:
        return scaler.transform(df_sub[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)

    train_X = to_X(df_train);  train_y = df_train["label"].astype(np.float32).values
    phase1_X = to_X(df_val);   phase1_y = df_val["label"].astype(int).values
    phase2_X = to_X(df_phase2);phase2_y = df_phase2["label"].astype(int).values
    phase2_fam = df_phase2["class_name"].values

    # Memory probe: balanced sample from val (original families, forgetting check)
    rng   = np.random.default_rng(SEED)
    r_idx = np.where(phase1_y == 1)[0]
    b_idx = np.where(phase1_y == 0)[0]
    r_sel = rng.choice(r_idx, size=min(PROBE_SIZE // 2, len(r_idx)), replace=False)
    b_sel = rng.choice(b_idx, size=min(PROBE_SIZE // 2, len(b_idx)), replace=False)
    probe_X = phase1_X[np.concatenate([r_sel, b_sel])]
    probe_y = phase1_y[np.concatenate([r_sel, b_sel])]
    print(f"  Memory probe: {len(probe_X)} samples (ransom={int((probe_y==1).sum())} benign={int((probe_y==0).sum())})")

    # ------------------------------------------------------------------ #
    # Warm-up: prequential eval on Phase 1 (no updates)
    # ------------------------------------------------------------------ #
    print("\nPhase 1 warm-up (no updates) …")
    model_ref = load_model(device)
    p1_probs  = score_batch(model_ref, phase1_X, device)
    p1_auprc  = float(average_precision_score(phase1_y, p1_probs))
    p1_auroc  = float(roc_auc_score(phase1_y, p1_probs))
    print(f"  Phase 1 AUPRC={p1_auprc:.4f}  AUROC={p1_auroc:.4f}  "
          f"(model before any Phase 2 update)")

    # ------------------------------------------------------------------ #
    # Run each strategy on Phase 2
    # ------------------------------------------------------------------ #
    strategies = ["no_update", "finetune", "replay", "ewc"]
    all_results = {"phase1": {"auprc": round(p1_auprc, 4), "auroc": round(p1_auroc, 4)}}

    print(f"\nPhase 2: {len(df_phase2):,} windows in {(len(df_phase2)+STEP_SIZE-1)//STEP_SIZE} steps "
          f"(step={STEP_SIZE}, budget={BUDGET})")
    print(f"{'='*70}")

    for strat in strategies:
        print(f"\n  Strategy: {strat}")
        result = run_strategy(
            strat, phase1_X, phase1_y,
            phase2_X, phase2_y, phase2_fam,
            probe_X, probe_y,
            train_X, train_y,
            scaler, device,
        )
        all_results[strat] = result
        rec = result["recovery_step"]
        print(f"    Overall AUPRC : {result['overall_auprc']:.4f}  "
              f"AUROC: {result['overall_auroc']:.4f}")
        print(f"    Forgetting    : {result['forgetting']:.4f}  "
              f"(probe AUPRC drop)")
        print(f"    OOD recall    : {result['ood_recall_overall']:.4f}  "
              f"({result['total_ood_windows']} OOD windows)")
        print(f"    Recovery step : {rec if rec is not None else 'never'}"
              + (f" / {result['n_steps']} steps" if rec is not None else ""))
        print(f"    FN before rec : {result['fn_before_recovery']}")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*85}")
    print("  E10 FEEDBACK LOOP SUMMARY")
    print(f"{'='*85}")
    print(f"  {'Strategy':<14} {'Phase2 AUPRC':>13} {'Phase2 AUROC':>13} "
          f"{'Forgetting':>11} {'OOD Recall':>11} {'Recovery':>9} {'FN/drift':>9}")
    print("  " + "-" * 83)
    for strat in strategies:
        r = all_results[strat]
        rec_s = str(r["recovery_step"]) if r["recovery_step"] is not None else "never"
        print(f"  {strat:<14} {r['overall_auprc']:>13.4f} {r['overall_auroc']:>13.4f} "
              f"{r['forgetting']:>11.4f} {r['ood_recall_overall']:>11.4f} "
              f"{rec_s:>9} {r['fn_before_recovery']:>9}")
    print(f"\n  Phase 1 baseline (no drift): AUPRC={p1_auprc:.4f}  AUROC={p1_auroc:.4f}")
    print(f"  OOD drift: Conti_01–07 variants, {len(ood_sorted)} windows, "
          f"Feb 26–29 2024")
    print(f"  Budget: {BUDGET} labels/step  |  Step size: {STEP_SIZE} windows  |  "
          f"EWC λ={EWC_LAMBDA}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "e10_feedback_loop.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    main()
