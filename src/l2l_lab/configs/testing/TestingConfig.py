from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from l2l_lab.configs.common.EnvConfig import EnvConfig
from l2l_lab.configs.testing.agents.AgentConfig import AgentConfig
from l2l_lab.configs.testing.agents.AlphaZeroMCTSAgentConfig import AlphaZeroMCTSAgentConfig
from l2l_lab.configs.testing.agents.PolicyAgentConfig import PolicyAgentConfig
from l2l_lab.configs.testing.agents.RandomAgentConfig import RandomAgentConfig
from l2l_lab.configs.testing.agents.TraditionalMCTSAgentConfig import TraditionalMCTSAgentConfig


def parse_agent_config(data: dict[str, Any]) -> AgentConfig:
    agent_type = data.get("agent_type", "random")
    if agent_type == "policy":
        return PolicyAgentConfig.from_dict(data.get("policy", {}))
    elif agent_type == "random":
        return RandomAgentConfig.from_dict()
    elif agent_type == "alphazero_mcts":
        return AlphaZeroMCTSAgentConfig.from_dict(data.get("alphazero_mcts", {}))
    elif agent_type == "traditional_mcts":
        return TraditionalMCTSAgentConfig.from_dict(data.get("traditional_mcts", {}))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


@dataclass
class TestingConfig:
    p1: AgentConfig
    p2: AgentConfig
    env: EnvConfig
    num_games: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestingConfig:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

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
