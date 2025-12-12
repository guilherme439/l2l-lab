from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from configs.definition.common.EnvConfig import EnvConfig
from configs.definition.testing.agents.AgentConfig import AgentConfig
from configs.definition.testing.agents.PolicyAgentConfig import PolicyAgentConfig
from configs.definition.testing.agents.RandomAgentConfig import RandomAgentConfig


def parse_agent_config(data: Dict[str, Any]) -> AgentConfig:
    agent_type = data.get("agent_type", "random")
    if agent_type == "policy":
        return PolicyAgentConfig.from_dict(data.get("policy", {}))
    elif agent_type == "random":
        return RandomAgentConfig.from_dict()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@dataclass
class TestingConfig:
    p1: AgentConfig
    p2: AgentConfig
    env: EnvConfig
    num_games: int = 1
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TestingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        agents_data = data.pop("agents", {})
        p1_config = parse_agent_config(agents_data.get("p1", {}))
        p2_config = parse_agent_config(agents_data.get("p2", {}))
        
        env_data = data.pop("env", {})
        env_config = EnvConfig(
            name=env_data.get("name"),
            obs_space_format=env_data.get("obs_space_format", "channels_first"),
            kwargs=env_data.get("kwargs", {}),
        )
        
        return cls(p1=p1_config, p2=p2_config, env=env_config, **data)
