from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import yaml


@dataclass
class TestingConfig:
    model_name: str
    checkpoint_1: str
    checkpoint_2: str
    game_config: str
    num_games: int = 1
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TestingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
