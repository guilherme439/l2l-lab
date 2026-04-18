from dataclasses import dataclass

from .AgentConfig import AgentConfig


@dataclass
class RandomAgentConfig(AgentConfig):
    
    def __post_init__(self):
        self.agent_type = "random"
    
    @classmethod
    def from_dict(cls) -> "RandomAgentConfig":
        return cls(agent_type="random")
