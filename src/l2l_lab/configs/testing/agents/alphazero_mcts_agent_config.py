from dataclasses import dataclass
from typing import Literal

from .agent_config import BaseAgentConfig


@dataclass
class AlphaZeroMCTSAgentConfig(BaseAgentConfig):
    agent_type: Literal["alphazero_mcts"] = "alphazero_mcts"
    model_name: str = ""
    checkpoint: int = 0
    is_recurrent: bool = False
    search_config_path: str = ""
