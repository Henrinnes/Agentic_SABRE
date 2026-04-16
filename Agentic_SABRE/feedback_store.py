# Agentic_SABRE/feedback_store.py
"""
Append-only JSON feedback log and numpy training-row exporter.

Layout of a feedback record
----------------------------
{
  "timestamp":            str  (ISO-8601),
  "sample_id":            str  (UUID assigned at inference time),
  "original_decision":    str  ("ALLOW" | "ESCALATE" | "AUTO_CONTAIN"),
  "p_sem":                float,
  "p_beh":                float,
  "p_know":               float,
  "sigma_sem":            float,
  "sigma_beh":            float,
  "sigma_know":           float,
  "fused_prob":           float,
  "expert_correction":    int   (0 = benign, 1 = ransomware),
  "expert_id":            str,
  "reasoning":            str,
  "consultation_triggered": {
      "semantic":   bool,
      "behavioral": bool,
      "knowledge":  bool
  }
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_LOG_PATH = Path("checkpoints/feedback/feedback_log.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_log(path: Path) -> List[Dict[str, Any]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_log(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def append_feedback(
    *,
    sample_id: str,
    original_decision: str,
    p_sem: float,
    p_beh: float,
    p_know: float,
    sigma_sem: float,
    sigma_beh: float,
    sigma_know: float,
    fused_prob: float,
    expert_correction: int,
    expert_id: str = "analyst",
    reasoning: str = "",
    consultation_triggered: Optional[Dict[str, bool]] = None,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> int:
    """
    Append one analyst correction to the feedback log.

    Returns the new total number of records in the log.
    """
    path = Path(log_path)
    records = _load_log(path)

    record: Dict[str, Any] = {
        "timestamp":             datetime.now(tz=timezone.utc).isoformat(),
        "sample_id":             sample_id,
        "original_decision":     original_decision,
        "p_sem":                 float(p_sem),
        "p_beh":                 float(p_beh),
        "p_know":                float(p_know),
        "sigma_sem":             float(sigma_sem),
        "sigma_beh":             float(sigma_beh),
        "sigma_know":            float(sigma_know),
        "fused_prob":            float(fused_prob),
        "expert_correction":     int(expert_correction),
        "expert_id":             str(expert_id),
        "reasoning":             str(reasoning),
        "consultation_triggered": consultation_triggered or {
            "semantic": False, "behavioral": False, "knowledge": False
        },
    }
    records.append(record)
    _save_log(records, path)
    return len(records)


def load_feedback(log_path: str | Path = DEFAULT_LOG_PATH) -> List[Dict[str, Any]]:
    """Return all records from the feedback log (empty list if absent)."""
    return _load_log(Path(log_path))


def feedback_count(log_path: str | Path = DEFAULT_LOG_PATH) -> int:
    """Return total number of feedback records without loading full data."""
    return len(_load_log(Path(log_path)))


def feedback_as_training_rows(
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert feedback log to (X, y) suitable for DecisionTrainer.

    X shape: [N, 6]  — columns: [p_sem, p_beh, p_know, sigma_sem, sigma_beh, sigma_know]
    y shape: [N]     — expert_correction (0 or 1)

    Returns
    -------
    X : np.ndarray, dtype=float32
    y : np.ndarray, dtype=float32
    """
    records = _load_log(Path(log_path))
    if not records:
        return np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.float32)

    rows = []
    labels = []
    for r in records:
        rows.append([
            float(r.get("p_sem",     0.5)),
            float(r.get("p_beh",     0.5)),
            float(r.get("p_know",    0.5)),
            float(r.get("sigma_sem", 0.0)),
            float(r.get("sigma_beh", 0.0)),
            float(r.get("sigma_know",0.0)),
        ])
        labels.append(float(r.get("expert_correction", 0)))

    X = np.array(rows,   dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y
