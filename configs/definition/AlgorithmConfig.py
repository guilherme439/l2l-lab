from dataclasses import dataclass
from typing import Union

from .algorithms.PPOConfig import AlgoPPOConfig
from .algorithms.IMPALAConfig import AlgoIMPALAConfig

AlgoConfigType = Union[AlgoPPOConfig, AlgoIMPALAConfig]

ALGO_CONFIG_MAP = {
    "ppo": AlgoPPOConfig,
    "impala": AlgoIMPALAConfig,
}


@dataclass
class AlgorithmConfig:
    name: str
    iterations: int
    config: AlgoConfigType = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        name = data.get("name")
        iterations = data.get("iterations")
        config_data = data.get("config", {}) or {}
        
        config_class = ALGO_CONFIG_MAP.get(name)
        if config_class is None:
            raise ValueError(f"Unknown algorithm: {name}. Supported: {list(ALGO_CONFIG_MAP.keys())}")
        
        config = config_class(**config_data)
        return cls(name=name, iterations=iterations, config=config)
