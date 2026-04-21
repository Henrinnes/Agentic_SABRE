# Agentic_SABRE/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from typing import List
from .agents.semantic import SemanticAgent
from .agents.behavioural import BehavioralAgent
from .agents.knowledge import KnowledgeAgent
from .agents.decision import DecisionAgent
from .contracts import AgentOutput

import json
from pathlib import Path

def load_auto_policy(path: str = "checkpoints/policy_auto.json"):
    blob = json.loads(Path(path).read_text())
    return blob
    
Action = Literal["ALLOW", "ESCALATE", "AUTO_CONTAIN"]


@dataclass
class TicketSink:
    system: str = "stdout"
    def create_ticket(self, payload: Dict[str, Any]) -> None:
        print(f"[TICKET->{self.system}] {payload}")


@dataclass
class ActionSink:
    system: str = "stdout"
    def isolate_host(self, host: str, reason: str) -> None:
        print(f"[ACTION->{self.system}] isolate_host host={host} reason={reason}")
    def kill_process(self, pid: int | None, reason: str) -> None:
        print(f"[ACTION->{self.system}] kill_process pid={pid} reason={reason}")


class Orchestrator:
    """
    Policy:
      - AUTO_CONTAIN if fused >= τ_high AND max_uncert <= κ_low (confident & high risk)
      - ESCALATE if fused >= τ OR max_uncert > κ (risk or uncertainty)
      - otherwise ALLOW
    """

    def __init__(
        self,
        τ: float = 0.80,
        κ: float = 0.15,
        τ_high: float | None = 0.95,
        κ_low: float | None = 0.08,
        decision_artifact: str | None = "checkpoints/decision_fuser_3agent.json",
    ):
        self.sem  = SemanticAgent()
        self.beh  = BehavioralAgent()
        self.know = KnowledgeAgent()
        self.dec  = DecisionAgent(
            artifact_path=decision_artifact,
            use_uncertainties=True,
        )

        self.τ = float(τ)
        self.κ = float(κ)
        self.τ_high = None if τ_high is None else float(τ_high)
        self.κ_low = None if κ_low is None else float(κ_low)

        self.tickets = TicketSink()
        self.actions = ActionSink()

    def _decide(self, fused_p: float, uncert_max: float) -> Action:
        if self.τ_high is not None and self.κ_low is not None:
            if fused_p >= self.τ_high and uncert_max <= self.κ_low:
                return "AUTO_CONTAIN"
        if fused_p >= self.τ or uncert_max > self.κ:
            return "ESCALATE"
        return "ALLOW"

    def handle(
        self,
        x_sem: Dict[str, Any],
        x_beh: Dict[str, Any],
        x_know: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        context = context or {}

        # 1) score with all three base agents (MC Dropout)
        o_sem:  AgentOutput = self.sem.score(x_sem)
        o_beh:  AgentOutput = self.beh.score(x_beh)
        o_know: AgentOutput = self.know.score(x_know)

        # 2) fuse all three scores with 3-agent decision MLP
        fused: AgentOutput = self.dec.score({
            "p_sem":  o_sem.prob,
            "p_beh":  o_beh.prob,
            "p_know": o_know.prob,
            "σ_sem":  o_sem.uncert,
            "σ_beh":  o_beh.uncert,
            "σ_know": o_know.uncert,
        })

        uncert_max = max(o_sem.uncert, o_beh.uncert, o_know.uncert)
        decision: Action = self._decide(fused.prob, uncert_max)

        # 3) side-effects (stubbed; replace with SOAR/ticketing)
        if decision == "AUTO_CONTAIN":
            pass
        elif decision == "ESCALATE":
            pass

        # 4) return rich result
        return {
            "decision": decision,
            "score_fused": fused.prob,
            "uncert_max": uncert_max,
            "by_agent": {
                "semantic":   o_sem.__dict__,
                "behavioral": o_beh.__dict__,
                "knowledge":  o_know.__dict__,
                "fusion":     fused.__dict__,
            },
            "context": context,
            "policy": {
                "tau": self.τ,
                "kappa": self.κ,
                "tau_high": self.τ_high,
                "kappa_low": self.κ_low,
            },
        }

    def handle_batch(
        self,
        x_sems:  List[Dict[str, Any]],
        x_behs:  List[Dict[str, Any]],
        x_knows: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference: processes B samples in parallel across all three agents.
        Semantically identical to calling handle() B times but faster on GPU.
        """
        B = len(x_sems)
        contexts = contexts or [{} for _ in range(B)]

        # 1) batch MC Dropout on all three base agents
        o_sems  = self.sem.score_batch(x_sems)
        o_behs  = self.beh.score_batch(x_behs)
        o_knows = self.know.score_batch(x_knows)

        # 2) batch 3-agent fusion
        fusion_inputs = [
            {"p_sem":  o_sems[i].prob,  "p_beh":  o_behs[i].prob,  "p_know": o_knows[i].prob,
             "σ_sem":  o_sems[i].uncert, "σ_beh": o_behs[i].uncert, "σ_know": o_knows[i].uncert}
            for i in range(B)
        ]
        fused_outputs = self.dec.score_batch(fusion_inputs)

        # 3) apply policy and build result dicts
        policy_info = {"tau": self.τ, "kappa": self.κ, "tau_high": self.τ_high, "kappa_low": self.κ_low}
        results = []
        for i in range(B):
            uncert_max = max(o_sems[i].uncert, o_behs[i].uncert, o_knows[i].uncert)
            decision = self._decide(fused_outputs[i].prob, uncert_max)
            results.append({
                "decision": decision,
                "score_fused": fused_outputs[i].prob,
                "uncert_max": uncert_max,
                "by_agent": {
                    "semantic":   o_sems[i].__dict__,
                    "behavioral": o_behs[i].__dict__,
                    "knowledge":  o_knows[i].__dict__,
                    "fusion":     fused_outputs[i].__dict__,
                },
                "context": contexts[i],
                "policy": policy_info,
            })
        return results