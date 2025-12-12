from dataclasses import dataclass
from typing import Optional, Union

from .algorithms.PPOConfig import AlgoPPOConfig
from .algorithms.IMPALAConfig import AlgoIMPALAConfig
from .PolicyConfig import PolicyConfig

AlgoConfigType = Union[AlgoPPOConfig, AlgoIMPALAConfig]

ALGO_CONFIG_MAP = {
    "ppo": AlgoPPOConfig,
    "impala": AlgoIMPALAConfig,
}


@dataclass
class AlgorithmConfig:
    name: str
    iterations: int
    config: Optional[AlgoConfigType] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        name = data.get("name")
        iterations = data.get("iterations")
        config_data = data.get("config", {}) or {}
        
        if name is None:
            raise ValueError("Algorithm name is required")
        if iterations is None:
            raise ValueError("Algorithm iterations is required")
        
        config_class = ALGO_CONFIG_MAP.get(name)
        if config_class is None:
            raise ValueError(f"Unknown algorithm: {name}. Supported: {list(ALGO_CONFIG_MAP.keys())}")
        
        policy_data = config_data.pop("policy", None)
        policy = PolicyConfig.from_dict(policy_data) if policy_data else None
        
        config = config_class(**config_data, policy=policy)
        return cls(name=name, iterations=iterations, config=config)
