#!/usr/bin/env python
"""
smoke_test_feedback.py
======================
Unit smoke tests for Feature 2: expert feedback loop.

No model files or real datasets required — uses synthetic data and
a temporary feedback log.

Run with:
    python smoke_test_feedback.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name, condition, msg=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {msg}")
        _failures.append(name)


# ---------------------------------------------------------------------------
# Test 1: append_feedback writes a valid record
# ---------------------------------------------------------------------------
def test_append_feedback():
    from Agentic_SABRE.feedback_store import append_feedback, load_feedback

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "feedback_log.json"

        n = append_feedback(
            sample_id="abc-123",
            original_decision="ESCALATE",
            p_sem=0.7, p_beh=0.6, p_know=0.8,
            sigma_sem=0.05, sigma_beh=0.04, sigma_know=0.06,
            fused_prob=0.75,
            expert_correction=1,
            expert_id="tester",
            reasoning="Test record",
            log_path=log_path,
        )
        check("append_returns_count_1", n == 1, f"n={n}")

        records = load_feedback(log_path)
        check("load_returns_list",  isinstance(records, list))
        check("load_has_1_record",  len(records) == 1)

        r = records[0]
        check("record_sample_id",   r["sample_id"] == "abc-123")
        check("record_decision",    r["original_decision"] == "ESCALATE")
        check("record_p_sem",       abs(r["p_sem"] - 0.7) < 1e-9)
        check("record_correction",  r["expert_correction"] == 1)
        check("record_timestamp",   "timestamp" in r and r["timestamp"])
        check("record_consultation_triggered",
              "consultation_triggered" in r)


# ---------------------------------------------------------------------------
# Test 2: Multiple appends accumulate correctly
# ---------------------------------------------------------------------------
def test_multiple_appends():
    from Agentic_SABRE.feedback_store import append_feedback, feedback_count

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "feedback_log.json"

        for i in range(5):
            n = append_feedback(
                sample_id=f"id-{i}",
                original_decision="ALLOW",
                p_sem=0.3, p_beh=0.3, p_know=0.3,
                sigma_sem=0.01, sigma_beh=0.01, sigma_know=0.01,
                fused_prob=0.3,
                expert_correction=0,
                log_path=log_path,
            )
        check("multiple_appends_count", n == 5, f"n={n}")
        check("feedback_count_fn", feedback_count(log_path) == 5)


# ---------------------------------------------------------------------------
# Test 3: feedback_as_training_rows returns correct shape
# ---------------------------------------------------------------------------
def test_training_rows_shape():
    from Agentic_SABRE.feedback_store import append_feedback, feedback_as_training_rows

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "feedback_log.json"

        for i in range(10):
            append_feedback(
                sample_id=f"id-{i}",
                original_decision="ESCALATE",
                p_sem=float(i)/10, p_beh=float(i)/10, p_know=float(i)/10,
                sigma_sem=0.1, sigma_beh=0.1, sigma_know=0.1,
                fused_prob=float(i)/10,
                expert_correction=i % 2,
                log_path=log_path,
            )

        X, y = feedback_as_training_rows(log_path)
        check("X_shape",   X.shape == (10, 6), f"shape={X.shape}")
        check("y_shape",   y.shape == (10,),   f"shape={y.shape}")
        check("X_dtype",   X.dtype == np.float32)
        check("y_dtype",   y.dtype == np.float32)
        check("y_binary",  set(y.astype(int).tolist()).issubset({0, 1}))


# ---------------------------------------------------------------------------
# Test 4: Empty log returns empty arrays
# ---------------------------------------------------------------------------
def test_empty_log():
    from Agentic_SABRE.feedback_store import feedback_as_training_rows, feedback_count

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nonexistent.json"
        X, y = feedback_as_training_rows(log_path)
        check("empty_X_shape",   X.shape == (0, 6), f"shape={X.shape}")
        check("empty_y_shape",   y.shape == (0,),   f"shape={y.shape}")
        check("empty_count",     feedback_count(log_path) == 0)


# ---------------------------------------------------------------------------
# Test 5: feedback_store creates parent directories
# ---------------------------------------------------------------------------
def test_creates_parent_dir():
    from Agentic_SABRE.feedback_store import append_feedback

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nested" / "deep" / "feedback_log.json"
        append_feedback(
            sample_id="x",
            original_decision="ALLOW",
            p_sem=0.1, p_beh=0.1, p_know=0.1,
            sigma_sem=0.0, sigma_beh=0.0, sigma_know=0.0,
            fused_prob=0.1,
            expert_correction=0,
            log_path=log_path,
        )
        check("nested_dir_created",  log_path.exists())


# ---------------------------------------------------------------------------
# Test 6: retrain_incremental with synthetic feedback (no real datasets)
# ---------------------------------------------------------------------------
def test_retrain_incremental_runs():
    base = "checkpoints/decision_fuser_3agent.json"
    if not Path(base).exists():
        print(f"  SKIP  retrain_incremental — base artifact not found: {base}")
        return

    from Agentic_SABRE.agents.decision import DecisionTrainer

    rng = np.random.default_rng(42)
    X_fb = rng.uniform(0, 1, (25, 6)).astype(np.float32)
    y_fb = rng.integers(0, 2, 25).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "updated.json")
        try:
            trainer = DecisionTrainer.retrain_incremental(
                base_artifact=base,
                X_feedback=X_fb,
                y_feedback=y_fb,
                out_path=out_path,
                feedback_repeat=2,
            )
            check("retrain_returns_trainer",
                  isinstance(trainer, DecisionTrainer))
            check("retrain_saved_artifact",
                  Path(out_path).exists())
            # Load back and do a forward pass
            loaded = DecisionTrainer.load(out_path)
            prob = loaded.model(
                __import__("torch").from_numpy(
                    loaded.scaler.transform(X_fb[:1]).astype("float32")
                ).to(getattr(loaded, "device", "cpu"))
            ).item()
            check("retrain_forward_pass",
                  0.0 <= prob <= 1.0, f"prob={prob}")
        except Exception as exc:
            check("retrain_no_exception", False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("append_feedback writes valid record",        test_append_feedback),
        ("Multiple appends accumulate correctly",      test_multiple_appends),
        ("feedback_as_training_rows shape",            test_training_rows_shape),
        ("Empty log returns empty arrays",             test_empty_log),
        ("feedback_store creates parent directories",  test_creates_parent_dir),
        ("retrain_incremental with synthetic data",    test_retrain_incremental_runs),
    ]

    print("=" * 60)
    print("smoke_test_feedback.py")
    print("=" * 60)
    for label, fn in tests:
        print(f"\n[{label}]")
        fn()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} test(s): {_failures}")
        sys.exit(1)
    else:
        print(f"ALL {sum(1 for _ in tests)} TEST GROUPS PASSED")
        sys.exit(0)
