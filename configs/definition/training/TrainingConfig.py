from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from .AlgorithmConfig import AlgorithmConfig
from .NetworkConfig import NetworkConfig
from configs.definition.common.EnvConfig import EnvConfig


@dataclass
class TrainingConfig:
    name: str
    env: EnvConfig
    algorithm: AlgorithmConfig
    network: NetworkConfig
    eval_interval: int = 20
    eval_games: int = 50
    eval_vs_previous: bool = False
    eval_graph_split: int = 500
    plot_interval: int = 50
    info_interval: int = 100
    checkpoint_interval: int = 100
    continue_training: bool = False
    continue_from_iteration: Optional[int] = None
    
    
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
        
        env_data = data.pop("env", {})
        env_config = EnvConfig(
            name=env_data.get("name"),
            obs_space_format=env_data.get("obs_space_format", "channels_first"),
            kwargs=env_data.get("kwargs", {}),
        )
        
        return cls(
            env=env_config,
            algorithm=algo_config,
            network=network_config,
            **data,
        )
