#!/usr/bin/env python
"""
smoke_test_ensemble.py
======================
Smoke tests for Deep Ensemble agents.

Uses synthetic models (random weights) so no real checkpoints or datasets
are required.

Run with:
    python smoke_test_ensemble.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import joblib

from Agentic_SABRE.contracts import AgentOutput
from Agentic_SABRE.agents.semantic    import CNNSemanticAgent
from Agentic_SABRE.agents.behavioural import CNNBehavioralAgent
from Agentic_SABRE.agents.knowledge   import CNNKnowledgeAgent

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name, condition, msg=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {msg}")
        _failures.append(name)


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_sem_checkpoints(tmpdir: Path, n: int = 5) -> Path:
    d = tmpdir / "semantic"; d.mkdir()
    for i in range(n):
        m = CNNSemanticAgent(input_dim=384, dropout=0.3)
        torch.save(m.state_dict(), str(d / f"seed_{i}.pt"))
    return d


def _make_beh_checkpoints(tmpdir: Path, n: int = 5) -> Path:
    d = tmpdir / "behavioral"; d.mkdir()
    for i in range(n):
        m = CNNBehavioralAgent(input_dim=16, dropout=0.3)
        torch.save(m.state_dict(), str(d / f"seed_{i}.pt"))
    return d


def _make_know_checkpoints(tmpdir: Path, n: int = 5) -> Path:
    d = tmpdir / "knowledge"; d.mkdir()
    device = torch.device("cpu")
    for i in range(n):
        m = CNNKnowledgeAgent(input_dim=23, dropout=0.3).to(device)
        with torch.no_grad():
            _ = m(torch.zeros(1, 1, 23))   # initialise LazyLinear
        torch.save(m.state_dict(), str(d / f"seed_{i}.pt"))
    # Fake scaler
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(np.random.rand(50, 23))
    joblib.dump(sc, str(d / "scaler.joblib"))
    return d


def _metadata(tmpdir: Path) -> str:
    import json
    p = tmpdir / "metadata.json"
    p.write_text(json.dumps({
        "semantic_input_dim": 384,
        "behavioral_input_dim": 16,
        "knowledge_input_dim": 23,
        "knowledge_features": [f"f{i}" for i in range(23)],
    }))
    return str(p)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_semantic_ensemble_score():
    from Agentic_SABRE.agents.ensemble import SemanticEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        ens_dir = _make_sem_checkpoints(tp)
        meta    = _metadata(tp)
        agent = SemanticEnsembleAgent(
            ensemble_dir=str(ens_dir), metadata_path=meta, n_members=5
        )
        x = {"embedding": np.random.rand(384).astype("float32")}
        out = agent.score(x)
        check("returns AgentOutput",        isinstance(out, AgentOutput))
        check("prob in [0,1]",              0.0 <= out.prob <= 1.0, f"prob={out.prob}")
        check("uncert >= 0",                out.uncert >= 0.0)
        check("member_probs length",        len(out.extras.get("member_probs", [])) == 5)
        check("note correct",               "ensemble" in out.extras.get("note", ""))


def test_behavioral_ensemble_score():
    from Agentic_SABRE.agents.ensemble import BehavioralEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        ens_dir = _make_beh_checkpoints(tp)
        meta    = _metadata(tp)
        agent = BehavioralEnsembleAgent(
            ensemble_dir=str(ens_dir), metadata_path=meta, n_members=5
        )
        x = {"features": np.random.rand(16).astype("float32")}
        out = agent.score(x)
        check("beh returns AgentOutput",    isinstance(out, AgentOutput))
        check("beh prob in [0,1]",          0.0 <= out.prob <= 1.0)
        check("beh uncert >= 0",            out.uncert >= 0.0)


def test_knowledge_ensemble_score():
    from Agentic_SABRE.agents.ensemble import KnowledgeEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        ens_dir = _make_know_checkpoints(tp)
        meta    = _metadata(tp)
        agent = KnowledgeEnsembleAgent(
            ensemble_dir=str(ens_dir), metadata_path=meta, n_members=5
        )
        x = {"features": np.random.rand(23).astype("float32")}
        out = agent.score(x)
        check("know returns AgentOutput",   isinstance(out, AgentOutput))
        check("know prob in [0,1]",         0.0 <= out.prob <= 1.0)
        check("know uncert >= 0",           out.uncert >= 0.0)


def test_score_batch_matches_single():
    """score_batch must produce same results as repeated score() calls."""
    from Agentic_SABRE.agents.ensemble import SemanticEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        ens_dir = _make_sem_checkpoints(tp, n=3)
        meta    = _metadata(tp)
        agent = SemanticEnsembleAgent(
            ensemble_dir=str(ens_dir), metadata_path=meta, n_members=3
        )
        rng = np.random.default_rng(7)
        inputs = [{"embedding": rng.random(384).astype("float32")} for _ in range(8)]

        singles = [agent.score(x) for x in inputs]
        batch   = agent.score_batch(inputs)

        for i in range(8):
            check(
                f"batch_prob_matches_single_i{i}",
                abs(batch[i].prob - singles[i].prob) < 1e-5,
                f"batch={batch[i].prob:.6f}  single={singles[i].prob:.6f}",
            )
            check(
                f"batch_uncert_matches_single_i{i}",
                abs(batch[i].uncert - singles[i].uncert) < 1e-5,
                f"batch={batch[i].uncert:.6f}  single={singles[i].uncert:.6f}",
            )


def test_uncert_nonzero_with_different_members():
    """
    With M=5 randomly initialised members, uncert should be > 0 for
    almost all inputs (degenerate collapse to identical outputs is
    astronomically unlikely with random weights).
    """
    from Agentic_SABRE.agents.ensemble import BehavioralEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        ens_dir = _make_beh_checkpoints(tp, n=5)
        meta    = _metadata(tp)
        agent = BehavioralEnsembleAgent(
            ensemble_dir=str(ens_dir), metadata_path=meta, n_members=5
        )
        rng = np.random.default_rng(42)
        inputs = [{"features": rng.random(16).astype("float32")} for _ in range(20)]
        outs = agent.score_batch(inputs)
        mean_uncert = np.mean([o.uncert for o in outs])
        check("uncert > 0 on avg (diverse members)", mean_uncert > 1e-6,
              f"mean_uncert={mean_uncert:.8f}")


def test_missing_checkpoint_raises():
    """Loading from non-existent directory raises FileNotFoundError."""
    from Agentic_SABRE.agents.ensemble import SemanticEnsembleAgent
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        meta = _metadata(tp)
        try:
            _ = SemanticEnsembleAgent(
                ensemble_dir=str(tp / "nonexistent"),
                metadata_path=meta, n_members=5,
            )
            check("missing_ckpt_raises", False, "Expected FileNotFoundError")
        except FileNotFoundError:
            check("missing_ckpt_raises", True)


def test_ece_helper():
    """ECE of a perfectly calibrated predictor should be near 0."""
    from eval_ensemble_vs_mcdropout import expected_calibration_error
    # Perfect calibration: confidence = frequency
    probs  = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 100)
    labels = (np.random.default_rng(0).random(len(probs)) < probs).astype(float)
    ece = expected_calibration_error(probs, labels, n_bins=9)
    check("ece_near_zero_for_calibrated", ece < 0.10, f"ece={ece:.4f}")


def test_brier_nll_helpers():
    from eval_ensemble_vs_mcdropout import brier, nll
    probs  = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    b = brier(probs, labels)
    n = nll(probs, labels)
    check("brier_perfect_preds_small", b < 0.05,  f"brier={b:.4f}")
    check("nll_perfect_preds_small",   n < 0.25,  f"nll={n:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("SemanticEnsembleAgent.score()",             test_semantic_ensemble_score),
        ("BehavioralEnsembleAgent.score()",           test_behavioral_ensemble_score),
        ("KnowledgeEnsembleAgent.score()",            test_knowledge_ensemble_score),
        ("score_batch matches repeated score()",      test_score_batch_matches_single),
        ("uncert > 0 with diverse members",           test_uncert_nonzero_with_different_members),
        ("missing checkpoint raises FileNotFoundError", test_missing_checkpoint_raises),
        ("ECE helper near 0 for calibrated model",   test_ece_helper),
        ("Brier / NLL helpers correct",               test_brier_nll_helpers),
    ]

    print("=" * 60)
    print("smoke_test_ensemble.py")
    print("=" * 60)
    for label, fn in tests:
        print(f"\n[{label}]")
        fn()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)}: {_failures}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TEST GROUPS PASSED")
        sys.exit(0)
