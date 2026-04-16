# Agentic_SABRE/orchestrator_ensemble.py
"""
Drop-in 3-agent orchestrator using Deep Ensemble agents.

Identical triage policy and consultation logic to orchestrator_3agent.py;
only the three expert agents are swapped from MC-Dropout to ensembles.

Usage
-----
  from Agentic_SABRE.orchestrator_ensemble import EnsembleOrchestrator
  orch = EnsembleOrchestrator(n_members=5)
  result = orch.handle(x_sem, x_beh, x_know)

The result dict is identical to Orchestrator.handle() with an added
'uq_method' key set to 'deep_ensemble'.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .agents.ensemble import (
    SemanticEnsembleAgent,
    BehavioralEnsembleAgent,
    KnowledgeEnsembleAgent,
)
from .agents.decision import DecisionAgent
from .contracts import AgentOutput
from .consultation import apply_consultation, apply_consultation_batch
from .orchestrator_3agent import _load_agent_policies, TicketSink, ActionSink

import json
from pathlib import Path

Action = Literal["ALLOW", "ESCALATE", "AUTO_CONTAIN"]


class EnsembleOrchestrator:
    """
    3-agent orchestrator with Deep Ensemble uncertainty estimation.

    All triage policy logic, consultation, and result structure are
    identical to the MC-Dropout Orchestrator — only the UQ source changes.
    """

    def __init__(
        self,
        τ: float = 0.80,
        κ: float = 0.15,
        τ_high: float | None = 0.95,
        κ_low:  float | None = 0.08,
        n_members: int = 5,
        ensemble_root: str = "checkpoints/ensemble",
        decision_artifact: str | None = "checkpoints/decision_fuser_3agent.json",
        agent_policies_path: str = "checkpoints/agent_policies.json",
    ):
        self.sem  = SemanticEnsembleAgent(
            ensemble_dir=f"{ensemble_root}/semantic",  n_members=n_members)
        self.beh  = BehavioralEnsembleAgent(
            ensemble_dir=f"{ensemble_root}/behavioral", n_members=n_members)
        self.know = KnowledgeEnsembleAgent(
            ensemble_dir=f"{ensemble_root}/knowledge",  n_members=n_members)

        self.dec = DecisionAgent(
            artifact_path=decision_artifact,
            use_uncertainties=True,
        )

        self.τ     = float(τ)
        self.κ     = float(κ)
        self.τ_high = None if τ_high is None else float(τ_high)
        self.κ_low  = None if κ_low  is None else float(κ_low)

        self.tickets = TicketSink()
        self.actions = ActionSink()
        self._agent_policies = _load_agent_policies(agent_policies_path)

    def _decide(self, fused_p: float, uncert_max: float) -> Action:
        if self.τ_high is not None and self.κ_low is not None:
            if fused_p >= self.τ_high and uncert_max <= self.κ_low:
                return "AUTO_CONTAIN"
        if fused_p >= self.τ or uncert_max > self.κ:
            return "ESCALATE"
        return "ALLOW"

    def _apply_consultation_single(self, o_sem, o_beh, o_know):
        outputs = {"semantic": o_sem, "behavioral": o_beh, "knowledge": o_know}
        updated, triggered = apply_consultation(outputs, self._agent_policies)
        return (updated["semantic"], updated["behavioral"],
                updated["knowledge"], triggered)

    def _apply_consultation_batch(self, o_sems, o_behs, o_knows):
        outputs_batch = [
            {"semantic": o_sems[i], "behavioral": o_behs[i], "knowledge": o_knows[i]}
            for i in range(len(o_sems))
        ]
        updated_batch, triggered_batch = apply_consultation_batch(
            outputs_batch, self._agent_policies
        )
        return (
            [updated_batch[i]["semantic"]   for i in range(len(o_sems))],
            [updated_batch[i]["behavioral"] for i in range(len(o_behs))],
            [updated_batch[i]["knowledge"]  for i in range(len(o_knows))],
            triggered_batch,
        )

    def handle(
        self,
        x_sem:   Dict[str, Any],
        x_beh:   Dict[str, Any],
        x_know:  Dict[str, Any],
        context: Dict[str, Any] | None = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context   = context or {}
        sample_id = sample_id or str(uuid.uuid4())

        o_sem:  AgentOutput = self.sem.score(x_sem)
        o_beh:  AgentOutput = self.beh.score(x_beh)
        o_know: AgentOutput = self.know.score(x_know)

        o_sem.sample_id  = sample_id
        o_beh.sample_id  = sample_id
        o_know.sample_id = sample_id

        o_sem, o_beh, o_know, consultation_triggered = \
            self._apply_consultation_single(o_sem, o_beh, o_know)

        fused: AgentOutput = self.dec.score({
            "p_sem":  o_sem.prob,   "p_beh":  o_beh.prob,   "p_know":  o_know.prob,
            "σ_sem":  o_sem.uncert, "σ_beh":  o_beh.uncert, "σ_know":  o_know.uncert,
        })

        uncert_max = max(o_sem.uncert, o_beh.uncert, o_know.uncert)
        decision: Action = self._decide(fused.prob, uncert_max)

        return {
            "decision":    decision,
            "score_fused": fused.prob,
            "uncert_max":  uncert_max,
            "sample_id":   sample_id,
            "uq_method":   "deep_ensemble",
            "by_agent": {
                "semantic":   o_sem.__dict__,
                "behavioral": o_beh.__dict__,
                "knowledge":  o_know.__dict__,
                "fusion":     fused.__dict__,
            },
            "context": context,
            "policy": {
                "tau": self.τ, "kappa": self.κ,
                "tau_high": self.τ_high, "kappa_low": self.κ_low,
            },
            "consultation_triggered": consultation_triggered,
        }

    def handle_batch(
        self,
        x_sems:   List[Dict[str, Any]],
        x_behs:   List[Dict[str, Any]],
        x_knows:  List[Dict[str, Any]],
        contexts: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        B = len(x_sems)
        contexts = contexts or [{} for _ in range(B)]

        o_sems  = self.sem.score_batch(x_sems)
        o_behs  = self.beh.score_batch(x_behs)
        o_knows = self.know.score_batch(x_knows)

        o_sems, o_behs, o_knows, triggered_batch = \
            self._apply_consultation_batch(o_sems, o_behs, o_knows)

        fusion_inputs = [
            {
                "p_sem":  o_sems[i].prob,   "p_beh":  o_behs[i].prob,   "p_know":  o_knows[i].prob,
                "σ_sem":  o_sems[i].uncert, "σ_beh":  o_behs[i].uncert, "σ_know":  o_knows[i].uncert,
            }
            for i in range(B)
        ]
        fused_outputs = self.dec.score_batch(fusion_inputs)

        results = []
        for i in range(B):
            uncert_max = max(o_sems[i].uncert, o_behs[i].uncert, o_knows[i].uncert)
            decision = self._decide(fused_outputs[i].prob, uncert_max)
            results.append({
                "decision":               decision,
                "score_fused":            fused_outputs[i].prob,
                "uncert_max":             uncert_max,
                "uq_method":              "deep_ensemble",
                "consultation_triggered": triggered_batch[i],
            })
        return results
