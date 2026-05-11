# Agentic_SABRE/orchestrator_3agent.py
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .agents.semantic import SemanticAgent
from .agents.behavioural import BehavioralAgent
from .agents.knowledge import KnowledgeAgent
from .agents.decision import DecisionAgent

from .contracts import AgentOutput
from .consultation import apply_consultation, apply_consultation_batch
from .feedback_store import append_feedback, feedback_as_training_rows
from .agents.decision import DecisionTrainer

import json
from pathlib import Path


def load_auto_policy(path: str = "checkpoints/policy_auto.json"):
    blob = json.loads(Path(path).read_text())
    return blob


def _load_agent_policies(path: str = "checkpoints/agent_policies.json") -> Dict[str, Dict[str, float]]:
    """
    Load per-agent policies.  Falls back to kappa_local=1.0 (consultation
    never fires) if the file is absent or unreadable.
    """
    default = {
        "semantic":   {"tau_local": 0.80, "kappa_local": 1.0},
        "behavioral": {"tau_local": 0.80, "kappa_local": 1.0},
        "knowledge":  {"tau_local": 0.80, "kappa_local": 1.0},
    }
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


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

    Per-agent consultation
    ----------------------
    Before fusion, if an agent's σ exceeds its per-agent κ_local threshold,
    the orchestrator adjusts that agent's probability using inverse-uncertainty-
    weighted peer scores.  Uncertainty values are never modified.
    """

    def __init__(
        self,
        τ: float = 0.80,
        κ: float = 0.15,
        τ_high: float | None = 0.95,
        κ_low: float | None = 0.08,
        decision_artifact: str | None = "checkpoints/decision_fuser_3agent.json",
        agent_policies_path: str = "checkpoints/agent_policies.json",
        feedback_log: str = "checkpoints/feedback/feedback_log.json",
        auto_retrain_threshold: int = 50,
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
        self.κ_low  = None if κ_low  is None else float(κ_low)

        self.tickets = TicketSink()
        self.actions = ActionSink()

        # per-agent policies (loaded once; fallback keeps consultation inactive)
        self._agent_policies = _load_agent_policies(agent_policies_path)

        # feedback store paths
        self._feedback_log = feedback_log
        self._decision_artifact = decision_artifact
        self._auto_retrain_threshold = auto_retrain_threshold

    def _decide(self, fused_p: float, uncert_max: float) -> Action:

        if self.τ_high is not None and self.κ_low is not None:
            if fused_p >= self.τ_high and uncert_max <= self.κ_low:
                return "AUTO_CONTAIN"

        if fused_p >= self.τ or uncert_max > self.κ:
            return "ESCALATE"

        return "ALLOW"

    # ------------------------------------------------------------------
    # Consultation helpers
    # ------------------------------------------------------------------

    def _apply_consultation_single(
        self,
        o_sem: AgentOutput,
        o_beh: AgentOutput,
        o_know: AgentOutput,
    ):
        """
        Apply per-agent consultation for a single sample.

        Returns
        -------
        (o_sem, o_beh, o_know, triggered_dict)
        """
        outputs = {
            "semantic":   o_sem,
            "behavioral": o_beh,
            "knowledge":  o_know,
        }
        updated, triggered = apply_consultation(outputs, self._agent_policies)
        return (
            updated["semantic"],
            updated["behavioral"],
            updated["knowledge"],
            triggered,
        )

    def _apply_consultation_batch(
        self,
        o_sems:  List[AgentOutput],
        o_behs:  List[AgentOutput],
        o_knows: List[AgentOutput],
    ):
        """
        Apply per-agent consultation for a batch of samples.

        Returns
        -------
        (o_sems, o_behs, o_knows, triggered_list)
        """
        outputs_batch = [
            {"semantic": o_sems[i], "behavioral": o_behs[i], "knowledge": o_knows[i]}
            for i in range(len(o_sems))
        ]
        updated_batch, triggered_batch = apply_consultation_batch(
            outputs_batch, self._agent_policies
        )
        new_sems  = [updated_batch[i]["semantic"]   for i in range(len(o_sems))]
        new_behs  = [updated_batch[i]["behavioral"] for i in range(len(o_behs))]
        new_knows = [updated_batch[i]["knowledge"]  for i in range(len(o_knows))]
        return new_sems, new_behs, new_knows, triggered_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # --------------------------------------------------
        # 1. Score with expert agents
        # --------------------------------------------------

        o_sem:  AgentOutput = self.sem.score(x_sem)
        o_beh:  AgentOutput = self.beh.score(x_beh)
        o_know: AgentOutput = self.know.score(x_know)

        # Attach sample_id so feedback can reference this sample
        o_sem.sample_id  = sample_id
        o_beh.sample_id  = sample_id
        o_know.sample_id = sample_id

        # --------------------------------------------------
        # 2. Per-agent consultation (adjusts probs if σ > κ_local)
        # --------------------------------------------------

        o_sem, o_beh, o_know, consultation_triggered = self._apply_consultation_single(
            o_sem, o_beh, o_know
        )

        # --------------------------------------------------
        # 3. Fusion (Decision Agent)
        # --------------------------------------------------

        fused: AgentOutput = self.dec.score({
            "p_sem":   o_sem.prob,
            "p_beh":   o_beh.prob,
            "p_know":  o_know.prob,
            "σ_sem":   o_sem.uncert,
            "σ_beh":   o_beh.uncert,
            "σ_know":  o_know.uncert,
        })

        uncert_max = max(o_sem.uncert, o_beh.uncert, o_know.uncert)

        decision: Action = self._decide(fused.prob, uncert_max)

        # --------------------------------------------------
        # 4. Side effects (SOAR / SOC integration)
        # --------------------------------------------------

        if decision == "AUTO_CONTAIN":
            self.actions.isolate_host(
                host=context.get("host", sample_id),
                reason=f"SABRE fused_p={fused.prob:.3f}",
            )

        elif decision == "ESCALATE":
            self.tickets.create_ticket({
                "sample_id": sample_id,
                "score_fused": fused.prob,
                "uncert_max": uncert_max,
                "by_agent": {
                    "p_sem": o_sem.prob,  "σ_sem": o_sem.uncert,
                    "p_beh": o_beh.prob,  "σ_beh": o_beh.uncert,
                    "p_know": o_know.prob, "σ_know": o_know.uncert,
                },
                "context": context,
            })

        # --------------------------------------------------
        # 5. Return rich result
        # --------------------------------------------------

        return {
            "decision":    decision,
            "score_fused": fused.prob,
            "uncert_max":  uncert_max,
            "sample_id":   sample_id,
            "by_agent": {
                "semantic":   o_sem.__dict__,
                "behavioral": o_beh.__dict__,
                "knowledge":  o_know.__dict__,
                "fusion":     fused.__dict__,
            },
            "context": context,
            "policy": {
                "tau":       self.τ,
                "kappa":     self.κ,
                "tau_high":  self.τ_high,
                "kappa_low": self.κ_low,
            },
            "consultation_triggered": consultation_triggered,
        }

    # ------------------------------------------------------------------
    # Analyst Feedback API
    # ------------------------------------------------------------------

    def record_analyst_correction(
        self,
        result: Dict[str, Any],
        expert_correction: int,
        expert_id: str = "analyst",
        reasoning: str = "",
    ) -> int:
        """
        Record an analyst correction for a previously ESCALATE'd sample.

        Parameters
        ----------
        result : dict
            The dict returned by `handle()` for the sample.
        expert_correction : int
            0 = analyst confirmed benign, 1 = analyst confirmed ransomware.
        expert_id : str
            Analyst identifier (for audit trail).
        reasoning : str
            Free-text analyst note.

        Returns
        -------
        int
            Total number of feedback records in the log after this append.
            If count reaches `auto_retrain_threshold`, the Decision MLP is
            automatically retrained and saved back to the artifact path.
        """
        by_agent = result.get("by_agent", {})
        sem   = by_agent.get("semantic",   {})
        beh   = by_agent.get("behavioral", {})
        know  = by_agent.get("knowledge",  {})
        fused = by_agent.get("fusion",     {})

        count = append_feedback(
            sample_id           = result.get("sample_id", ""),
            original_decision   = result.get("decision", "ESCALATE"),
            p_sem               = sem.get("prob", 0.5),
            p_beh               = beh.get("prob", 0.5),
            p_know              = know.get("prob", 0.5),
            sigma_sem           = sem.get("uncert", 0.0),
            sigma_beh           = beh.get("uncert", 0.0),
            sigma_know          = know.get("uncert", 0.0),
            fused_prob          = fused.get("prob", result.get("score_fused", 0.5)),
            expert_correction   = expert_correction,
            expert_id           = expert_id,
            reasoning           = reasoning,
            consultation_triggered = result.get("consultation_triggered"),
            log_path            = self._feedback_log,
        )

        # Trigger incremental Decision MLP retrain when threshold is reached
        if (
            count >= self._auto_retrain_threshold
            and count % self._auto_retrain_threshold == 0
            and self._decision_artifact is not None
            and Path(self._decision_artifact).exists()
        ):
            X_fb, y_fb = feedback_as_training_rows(self._feedback_log)
            if len(X_fb) >= 10:
                updated = DecisionTrainer.retrain_incremental(
                    base_artifact = self._decision_artifact,
                    X_feedback    = X_fb,
                    y_feedback    = y_fb,
                    out_path      = self._decision_artifact,
                )
                # Reload the decision agent with the updated artifact
                self.dec.trainer = DecisionTrainer.load(self._decision_artifact)
                print(
                    f"[Orchestrator] Decision MLP retrained on {len(X_fb)} "
                    f"feedback rows → {self._decision_artifact}"
                )

        return count

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

        # Per-agent consultation (vectorised)
        o_sems, o_behs, o_knows, triggered_batch = self._apply_consultation_batch(
            o_sems, o_behs, o_knows
        )

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
                "consultation_triggered": triggered_batch[i],
            })
        return results
