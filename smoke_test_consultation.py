#!/usr/bin/env python
"""
smoke_test_consultation.py
==========================
Unit smoke tests for Feature 1: per-agent consultation engine.

All tests use synthetic AgentOutput objects — no model files required.

Run with:
    python smoke_test_consultation.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

from Agentic_SABRE.contracts import AgentOutput
from Agentic_SABRE.consultation import (
    AGENT_NAMES,
    apply_consultation,
    apply_consultation_batch,
    consult_single,
)


def _make_outputs(probs, sigmas):
    names = AGENT_NAMES
    return {n: AgentOutput(prob=p, uncert=s) for n, p, s in zip(names, probs, sigmas)}


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
# Test 1: No consultation when all σ ≤ κ_local
# ---------------------------------------------------------------------------
def test_no_consultation_below_threshold():
    outputs = _make_outputs([0.7, 0.6, 0.8], [0.05, 0.04, 0.06])
    policies = {n: {"kappa_local": 0.10} for n in AGENT_NAMES}
    updated, triggered = apply_consultation(outputs, policies)

    check("no_consult_triggered_when_below",
          not any(triggered.values()),
          f"triggered={triggered}")
    check("no_consult_probs_unchanged",
          all(updated[n].prob == outputs[n].prob for n in AGENT_NAMES))
    check("no_consult_consulted_false",
          all(not updated[n].consulted for n in AGENT_NAMES))


# ---------------------------------------------------------------------------
# Test 2: Consultation fires for one agent when σ > κ_local
# ---------------------------------------------------------------------------
def test_consultation_single_agent():
    # Only semantic is uncertain
    outputs = _make_outputs([0.7, 0.6, 0.8], [0.30, 0.04, 0.06])
    policies = {
        "semantic":   {"kappa_local": 0.20},
        "behavioral": {"kappa_local": 0.20},
        "knowledge":  {"kappa_local": 0.20},
    }
    updated, triggered = apply_consultation(outputs, policies)

    check("single_agent_triggered_semantic", triggered["semantic"])
    check("single_agent_not_triggered_behavioral", not triggered["behavioral"])
    check("single_agent_not_triggered_knowledge",  not triggered["knowledge"])
    check("single_agent_consulted_flag",  updated["semantic"].consulted)
    check("single_agent_peers_populated",
          len(updated["semantic"].consultation_peers) == 2)

    # Uncertainty must not change
    check("uncert_unchanged",
          updated["semantic"].uncert == outputs["semantic"].uncert)

    # Adjusted prob should differ from original (unless all probs equal)
    p_orig = outputs["semantic"].prob
    p_adj  = updated["semantic"].prob
    check("prob_adjusted", abs(p_adj - p_orig) > 1e-9,
          f"orig={p_orig:.4f}  adj={p_adj:.4f}")

    # raw_prob stored in extras
    check("raw_prob_in_extras",
          "raw_prob" in updated["semantic"].extras)
    check("raw_prob_correct",
          updated["semantic"].extras["raw_prob"] == p_orig)


# ---------------------------------------------------------------------------
# Test 3: Weighted mean formula correctness
# ---------------------------------------------------------------------------
def test_weighted_mean_correctness():
    # All same sigma → adjusted should be unweighted mean
    p  = np.array([0.3, 0.6, 0.9])
    s  = np.array([0.5, 0.5, 0.5])  # same sigma for all
    outputs = _make_outputs(p.tolist(), s.tolist())
    policies = {n: {"kappa_local": 0.1} for n in AGENT_NAMES}

    updated, _ = apply_consultation(outputs, policies)

    for agent_name in AGENT_NAMES:
        expected = float(np.mean(p))   # equal weights → arithmetic mean
        actual   = updated[agent_name].prob
        check(f"weighted_mean_{agent_name}",
              abs(actual - expected) < 1e-6,
              f"expected={expected:.4f}  got={actual:.4f}")


# ---------------------------------------------------------------------------
# Test 4: kappa_local=1.0 (default) → consultation never fires
# ---------------------------------------------------------------------------
def test_default_policy_no_consultation():
    # Very high uncertainty, but kappa_local=1.0 means never trigger
    outputs = _make_outputs([0.5, 0.5, 0.5], [0.99, 0.99, 0.99])
    policies = {n: {"kappa_local": 1.0} for n in AGENT_NAMES}
    updated, triggered = apply_consultation(outputs, policies)

    check("default_policy_never_triggers",
          not any(triggered.values()), f"triggered={triggered}")


# ---------------------------------------------------------------------------
# Test 5: Batch consultation produces same results as single
# ---------------------------------------------------------------------------
def test_batch_matches_single():
    rng = np.random.default_rng(0)
    B = 10
    all_outputs = [
        _make_outputs(rng.uniform(0.1, 0.9, 3).tolist(),
                      rng.uniform(0.0, 0.4, 3).tolist())
        for _ in range(B)
    ]
    policies = {n: {"kappa_local": 0.20} for n in AGENT_NAMES}

    single_results = [apply_consultation(o, policies) for o in all_outputs]
    batch_upd, batch_trig = apply_consultation_batch(all_outputs, policies)

    for i in range(B):
        s_upd, s_trig = single_results[i]
        for name in AGENT_NAMES:
            check(
                f"batch_matches_single_i{i}_{name}_prob",
                abs(batch_upd[i][name].prob - s_upd[name].prob) < 1e-9,
                f"batch={batch_upd[i][name].prob:.6f}  single={s_upd[name].prob:.6f}",
            )
            check(
                f"batch_matches_single_i{i}_{name}_triggered",
                batch_trig[i][name] == s_trig[name],
            )


# ---------------------------------------------------------------------------
# Test 6: AgentOutput backward compatibility — new fields have safe defaults
# ---------------------------------------------------------------------------
def test_agent_output_backward_compat():
    o = AgentOutput(prob=0.7, uncert=0.1)
    check("default_consulted_false",         not o.consulted)
    check("default_consultation_peers_list", o.consultation_peers == [])
    check("default_sample_id_none",          o.sample_id is None)


# ---------------------------------------------------------------------------
# Test 7: Orchestrator result contains consultation_triggered key
# ---------------------------------------------------------------------------
def test_orchestrator_has_consultation_triggered():
    try:
        from Agentic_SABRE.orchestrator_3agent import Orchestrator
    except Exception as exc:
        check("orchestrator_import", False, str(exc))
        return

    try:
        orch = Orchestrator()
    except Exception as exc:
        check("orchestrator_init", False, str(exc))
        return

    result = orch.handle(
        x_sem  = {"embedding": np.random.rand(384).astype("float32")},
        x_beh  = {"features":  np.random.rand(16).astype("float32")},
        x_know = {"features":  np.random.rand(23).astype("float32")},
    )
    check("result_has_consultation_triggered",
          "consultation_triggered" in result,
          f"keys={list(result.keys())}")
    check("result_has_sample_id",
          "sample_id" in result)
    ct = result["consultation_triggered"]
    check("consultation_triggered_has_all_agents",
          all(k in ct for k in ("semantic", "behavioral", "knowledge")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("No consultation below threshold",    test_no_consultation_below_threshold),
        ("Consultation single agent",          test_consultation_single_agent),
        ("Weighted mean correctness",          test_weighted_mean_correctness),
        ("Default policy no consultation",     test_default_policy_no_consultation),
        ("Batch matches single",               test_batch_matches_single),
        ("AgentOutput backward compat",        test_agent_output_backward_compat),
        ("Orchestrator consultation_triggered",test_orchestrator_has_consultation_triggered),
    ]

    print("=" * 60)
    print("smoke_test_consultation.py")
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
