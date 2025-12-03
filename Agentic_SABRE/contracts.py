# Agentic_SABRE/contracts.py
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

@dataclass
class AgentOutput:
    prob: float
    uncert: float
    extras: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(Protocol):
    name: str
    def score(self, x: Dict[str, Any]) -> AgentOutput:
        ...

