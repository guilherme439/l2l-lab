from dataclasses import dataclass
from typing import Literal

from .agent_config import BaseAgentConfig


@dataclass
class PolicyAgentConfig(BaseAgentConfig):
    agent_type: Literal["policy"] = "policy"
    model_name: str = ""
    checkpoint: int = 0
