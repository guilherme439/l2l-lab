from dataclasses import dataclass
from typing import Literal

from .agent_config import BaseAgentConfig


@dataclass
class RandomAgentConfig(BaseAgentConfig):
    agent_type: Literal["random"] = "random"
