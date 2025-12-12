from dataclasses import dataclass
from typing import Literal


AgentType = Literal["policy", "random"]


@dataclass
class AgentConfig:
    agent_type: AgentType
