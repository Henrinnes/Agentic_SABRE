from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

@dataclass
class AgentOutput:
    prob: float                    # mean risk in [0,1]
    uncert: float                  # predictive std (uncertainty)
    extras: Dict[str, Any] = field(default_factory=dict)  # e.g., SHAP, inputs, notes

class BaseAgent(Protocol):
    name: str
    def score(self, x: Dict[str, Any]) -> AgentOutput: ...

