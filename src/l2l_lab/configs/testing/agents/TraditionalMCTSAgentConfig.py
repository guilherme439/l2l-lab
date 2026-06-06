from dataclasses import dataclass
from typing import Any

from .AgentConfig import AgentConfig


@dataclass
class TraditionalMCTSAgentConfig(AgentConfig):
    search_config_path: str = ""

    def __post_init__(self):
        self.agent_type = "traditional_mcts"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraditionalMCTSAgentConfig:
        return cls(
            agent_type="traditional_mcts",
            search_config_path=data.get("search_config_path", ""),
        )
