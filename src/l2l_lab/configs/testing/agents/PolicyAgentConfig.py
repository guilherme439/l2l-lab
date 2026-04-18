from dataclasses import dataclass
from typing import Any, Dict

from .AgentConfig import AgentConfig


@dataclass
class PolicyAgentConfig(AgentConfig):
    model_name: str = ""
    checkpoint: int = 0
    
    def __post_init__(self):
        self.agent_type = "policy"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyAgentConfig":
        return cls(
            agent_type="policy",
            model_name=data.get("model_name", ""),
            checkpoint=data.get("checkpoint", 0),
        )
