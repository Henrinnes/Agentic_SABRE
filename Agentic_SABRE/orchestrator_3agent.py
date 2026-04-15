# Agentic_SABRE/orchestrator_3agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

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
    SABRE decision policy

      AUTO_CONTAIN  if fused >= τ_high AND uncertainty <= κ_low
      ESCALATE      if fused >= τ OR uncertainty > κ
      ALLOW         otherwise
    """

    def __init__(
        self,
        τ: float = 0.80,
        κ: float = 0.15,
        τ_high: float | None = 0.95,
        κ_low: float | None = 0.08,
        decision_artifact: str | None = "checkpoints/decision_fuser_3agent.json",
    ):

        # --- expert agents ---
        self.sem = SemanticAgent()
        self.beh = BehavioralAgent()
        self.know = KnowledgeAgent()

        # --- decision fusion agent ---
        self.dec = DecisionAgent(
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

        # --------------------------------------------------
        # 1. Score with expert agents
        # --------------------------------------------------

        o_sem: AgentOutput = self.sem.score(x_sem)
        o_beh: AgentOutput = self.beh.score(x_beh)
        o_know: AgentOutput = self.know.score(x_know)

        # --------------------------------------------------
        # 2. Fusion (Decision Agent)
        # --------------------------------------------------

        fused: AgentOutput = self.dec.score({
            "p_sem": o_sem.prob,
            "p_beh": o_beh.prob,
            "p_know": o_know.prob,
            "σ_sem": o_sem.uncert,
            "σ_beh": o_beh.uncert,
            "σ_know": o_know.uncert,
        })

        uncert_max = max(o_sem.uncert, o_beh.uncert, o_know.uncert)

        decision: Action = self._decide(fused.prob, uncert_max)

        # --------------------------------------------------
        # 3. Side effects (SOAR / SOC integration)
        # --------------------------------------------------

        if decision == "AUTO_CONTAIN":

            # self.actions.kill_process(
            #     pid=context.get("pid"),
            #     reason="High ransomware risk (low uncertainty)"
            # )

            # self.actions.isolate_host(
            #     host=context.get("host", "unknown"),
            #     reason="High ransomware risk (low uncertainty)"
            # )

            pass

        elif decision == "ESCALATE":

            # self.tickets.create_ticket({
            #     "title": "Ransomware suspicion",
            #     "score": fused.prob,
            #     "uncertainty": uncert_max,
            #     "context": context,
            #     "explanations": {
            #         "semantic": o_sem.extras,
            #         "behavioral": o_beh.extras,
            #         "knowledge": o_know.extras,
            #         "fusion": fused.extras,
            #     },
            # })

            pass

        # --------------------------------------------------
        # 4. Return rich result
        # --------------------------------------------------

        return {
            "decision": decision,
            "score_fused": fused.prob,
            "uncert_max": uncert_max,
            "by_agent": {
                "semantic": o_sem.__dict__,
                "behavioral": o_beh.__dict__,
                "knowledge": o_know.__dict__,
                "fusion": fused.__dict__,
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
        x_sems: List[Dict[str, Any]],
        x_behs: List[Dict[str, Any]],
        x_knows: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        B = len(x_sems)
        contexts = contexts or [{} for _ in range(B)]

        o_sems  = self.sem.score_batch(x_sems)
        o_behs  = self.beh.score_batch(x_behs)
        o_knows = self.know.score_batch(x_knows)

        fusion_inputs = [
            {
                "p_sem":   o_sems[i].prob,   "p_beh":   o_behs[i].prob,   "p_know":   o_knows[i].prob,
                "σ_sem":   o_sems[i].uncert, "σ_beh":   o_behs[i].uncert, "σ_know":   o_knows[i].uncert,
            }
            for i in range(B)
        ]
        fused_outputs = self.dec.score_batch(fusion_inputs)

        results = []
        for i in range(B):
            uncert_max = max(o_sems[i].uncert, o_behs[i].uncert, o_knows[i].uncert)
            decision = self._decide(fused_outputs[i].prob, uncert_max)
            results.append({
                "decision":    decision,
                "score_fused": fused_outputs[i].prob,
                "uncert_max":  uncert_max,
            })
        return results