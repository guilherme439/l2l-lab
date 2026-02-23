from dataclasses import dataclass
from typing import Optional, Union

from ..utils import dataclass_from_dict
from .algorithms.PPOConfig import AlgoPPOConfig
from .algorithms.IMPALAConfig import AlgoIMPALAConfig

ALGO_CONFIG_MAP = {
    "ppo": AlgoPPOConfig,
    "impala": AlgoIMPALAConfig,
}

try:
    from alphazoo import AlphaZooConfig
    ALGO_CONFIG_MAP["alphazero"] = AlphaZooConfig
    AlgoConfigType = Union[AlgoPPOConfig, AlgoIMPALAConfig, AlphaZooConfig]
except ImportError:
    AlgoConfigType = Union[AlgoPPOConfig, AlgoIMPALAConfig]


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

        config = dataclass_from_dict(config_class, config_data)

        return cls(name=name, iterations=iterations, config=config)
