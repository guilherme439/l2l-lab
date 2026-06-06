from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import TypeAdapter

from l2l_lab.configs.common.env_config import EnvConfig
from l2l_lab.configs.testing.agents import AgentConfig


@dataclass
class TestingConfig:
    p1: AgentConfig
    p2: AgentConfig
    env: EnvConfig
    num_games: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> TestingConfig:
        return cls.from_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestingConfig:
        return TypeAdapter(cls).validate_python(data)
