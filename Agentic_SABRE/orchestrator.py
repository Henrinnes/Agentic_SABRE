# Agentic_SABRE/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from .agents.semantic import SemanticAgent
from .agents.behavioural import BehavioralAgent
from .agents.decision import DecisionAgent
from .contracts import AgentOutput

Action = Literal["ALLOW", "ESCALATE", "AUTO_CONTAIN"]


# --- simple sinks (replace with your real integrations later) ---
@dataclass
class TicketSink:
    system: str = "stdout"
    def create_ticket(self, payload: Dict[str, Any]) -> None:
        # TODO: replace with real ticketing (TheHive/Jira/ServiceNow/etc.)
        print(f"[TICKET->{self.system}] {payload}")

@dataclass
class ActionSink:
    system: str = "stdout"
    def isolate_host(self, host: str, reason: str) -> None:
        # TODO: replace with real SOAR/EDR action
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
    ):
        self.sem = SemanticAgent()
        self.beh = BehavioralAgent()
        self.dec = DecisionAgent()

        self.τ = float(τ)
        self.κ = float(κ)
        self.τ_high = None if τ_high is None else float(τ_high)
        self.κ_low = None if κ_low is None else float(κ_low)

        # sinks (swap these with real integrations)
        self.tickets = TicketSink()
        self.actions = ActionSink()

    # ---- decision policy ----------------------------------------------------
    def _decide(self, fused_p: float, uncert_max: float) -> Action:
        if self.τ_high is not None and self.κ_low is not None:
            if fused_p >= self.τ_high and uncert_max <= self.κ_low:
                return "AUTO_CONTAIN"
        if fused_p >= self.τ or uncert_max > self.κ:
            return "ESCALATE"
        return "ALLOW"

    # ---- main entry ---------------------------------------------------------
    def handle(
        self,
        x_sem: Dict[str, Any],
        x_beh: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        context = context or {}

        # 1) score with agents
        o_sem: AgentOutput = self.sem.score(x_sem)
        o_beh: AgentOutput = self.beh.score(x_beh)
        fused: AgentOutput = self.dec.score({"p_sem": o_sem.prob, "p_beh": o_beh.prob})

        # 2) apply policy
        uncert_max = max(o_sem.uncert, o_beh.uncert)
        decision: Action = self._decide(fused.prob, uncert_max)

        # 3) route actions (your requested snippet, expanded)
        if decision == "AUTO_CONTAIN":
            # e.g., call your SOAR/EDR wrappers
            # self.actions.kill_process(pid=context.get("pid"), reason="High risk, low uncertainty")
            # self.actions.isolate_host(host=context.get("host", "unknown"), reason="High risk, low uncertainty")
            pass
        elif decision == "ESCALATE":
            # self.tickets.create_ticket({
            #     "title": "Ransomware suspicion",
            #     "score": fused.prob,
            #     "uncert_max": uncert_max,
            #     "context": context,
            #     "explanations": {
            #         "semantic": o_sem.extras,
            #         "behavioral": o_beh.extras,
            #         "fusion": fused.extras,
            #     },
            # })
            pass

        # 4) return rich result (for logs/CLI/UI)
        return {
            "decision": decision,
            "score_fused": fused.prob,
            "uncert_max": uncert_max,
            "by_agent": {
                "semantic": o_sem.__dict__,
                "behavioral": o_beh.__dict__,
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
