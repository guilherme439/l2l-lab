from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import yaml

from .algorithms.PPOConfig import PPOConfig
from .NetworkConfig import NetworkConfig


@dataclass
class TrainingConfig:
    name: str
    iterations: int
    algorithm: Literal["ppo"] = "ppo"
    algorithm_config: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    game_config: str = ""
    eval_interval: int = 20
    eval_games: int = 50
    plot_interval: int = 50
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        algo_data = data.pop("algorithm_config", {})
        algo_config = PPOConfig(**algo_data)
        
        network_data = data.pop("network", {})
        network_config = NetworkConfig(**network_data)
        
        return cls(
            algorithm_config=algo_config,
            network=network_config,
            **data,
        )
