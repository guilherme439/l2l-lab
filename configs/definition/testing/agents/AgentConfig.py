from dataclasses import dataclass
from typing import Literal


AgentType = Literal["policy", "random", "rl_module"]


@dataclass
class AgentConfig:
    agent_type: AgentType
