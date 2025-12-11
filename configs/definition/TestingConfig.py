from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import yaml

from .EnvConfig import EnvConfig


@dataclass
class TestingConfig:
    model_name: str
    checkpoint_1: str
    checkpoint_2: str
    env: EnvConfig
    num_games: int = 1
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TestingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        env_data = data.pop("env", {})
        env_config = EnvConfig(
            name=env_data.get("name"),
            obs_space_format=env_data.get("obs_space_format", "channels_first"),
            kwargs=env_data.get("kwargs", {}),
        )
        
        return cls(env=env_config, **data)
