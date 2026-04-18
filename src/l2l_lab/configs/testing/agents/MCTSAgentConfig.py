from dataclasses import dataclass
from typing import Any, Dict

from .AgentConfig import AgentConfig


@dataclass
class MCTSAgentConfig(AgentConfig):
    model_name: str = ""
    checkpoint: int = 0
    is_recurrent: bool = False
    search_config_path: str = ""

    def __post_init__(self):
        self.agent_type = "mcts"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSAgentConfig":
        return cls(
            agent_type="mcts",
            model_name=data.get("model_name", ""),
            checkpoint=data.get("checkpoint", 0),
            is_recurrent=data.get("is_recurrent", False),
            search_config_path=data.get("search_config_path", ""),
        )
