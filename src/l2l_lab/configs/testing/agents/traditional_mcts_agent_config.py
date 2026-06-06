from dataclasses import dataclass
from typing import Literal

from .agent_config import BaseAgentConfig


@dataclass
class TraditionalMCTSAgentConfig(BaseAgentConfig):
    agent_type: Literal["traditional_mcts"] = "traditional_mcts"
    search_config_path: str = ""
