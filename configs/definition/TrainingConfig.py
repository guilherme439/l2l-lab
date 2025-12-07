from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

from .AlgorithmConfig import AlgorithmConfig
from .NetworkConfig import NetworkConfig


@dataclass
class TrainingConfig:
    name: str
    algorithm: AlgorithmConfig
    network: NetworkConfig
    game_config: str = ""
    eval_interval: int = 20
    eval_games: int = 50
    eval_vs_previous: bool = False
    plot_interval: int = 50
    checkpoint_interval: int = 100
    continue_training: bool = False
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        algo_data = data.pop("algorithm", {})
        algo_config = AlgorithmConfig.from_dict(algo_data)
        
        network_data = data.pop("network", {})
        network_config = NetworkConfig(
            architecture=network_data.pop("architecture"),
            kwargs=network_data,
        )
        
        return cls(
            algorithm=algo_config,
            network=network_config,
            **data,
        )
