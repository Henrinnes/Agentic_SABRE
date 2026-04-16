# # Agentic_SABRE/contracts.py
# from dataclasses import dataclass, field
# from typing import Any, Dict, Protocol

# @dataclass
# class AgentOutput:
#     prob: float
#     uncert: float
#     extras: Dict[str, Any] = field(default_factory=dict)

# class BaseAgent(Protocol):
#     name: str
#     def score(self, x: Dict[str, Any]) -> AgentOutput:
#         ...

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Optional


@dataclass
class AgentOutput:
    """
    Standard output format for all SABRE agents.

    prob:
        Probability that the sample is ransomware-related.

    uncert:
        Uncertainty estimate produced by the agent
        (e.g., MC-Dropout std or entropy).

    extras:
        Optional dictionary for diagnostics or explanations
        (e.g., top features, model notes, intermediate signals).

    consulted:
        True if this agent's probability was adjusted by peer consultation.

    consultation_peers:
        Names of peer agents whose scores were used during consultation.

    sample_id:
        Optional UUID assigned by the orchestrator for feedback tracking.
    """
    prob: float
    uncert: float
    extras: Dict[str, Any] = field(default_factory=dict)
    consulted: bool = False
    consultation_peers: List[str] = field(default_factory=list)
    sample_id: Optional[str] = None


@dataclass
class FusionInput:
    """
    Standard structure passed to the Decision Agent.

    Contains outputs from all expert agents.
    """
    p_sem: float
    p_beh: float
    p_know: float

    sigma_sem: float = 0.0
    sigma_beh: float = 0.0
    sigma_know: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionOutput:
    """
    Final decision layer output.

    prob:
        Final fused ransomware probability.

    uncert:
        Fusion uncertainty used for SOC escalation.

    decision:
        Optional categorical label (e.g., benign/ransomware/escalate).

    extras:
        Debug information or model explanations.
    """
    prob: float
    uncert: float
    decision: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(Protocol):
    """
    Interface implemented by all expert agents
    (semantic, behavioural, knowledge).
    """

    name: str

    def score(self, x: Dict[str, Any]) -> AgentOutput:
        """
        Score a single input sample.

        Parameters
        ----------
        x : dict
            Input data specific to the agent
            (e.g., features, embeddings, behavioural metrics)

        Returns
        -------
        AgentOutput
            Probability, uncertainty, and optional metadata.
        """
        ...


class DecisionAgentInterface(Protocol):
    """
    Interface for the SABRE decision/fusion agent.
    """

    name: str

    def score(self, x: FusionInput) -> FusionOutput:
        """
        Fuse multiple agent predictions into a final decision.
        """
        ...