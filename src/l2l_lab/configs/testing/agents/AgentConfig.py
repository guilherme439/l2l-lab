from dataclasses import dataclass
from typing import Literal


AgentType = Literal["policy", "random", "rl_module", "alphazero_mcts", "traditional_mcts"]


@dataclass
class AgentConfig:
    agent_type: AgentType
