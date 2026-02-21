from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from .algorithms.PPOConfig import AlgoPPOConfig
from .algorithms.IMPALAConfig import AlgoIMPALAConfig
from .PolicyConfig import PolicyConfig

AlgoConfigType = Union[AlgoPPOConfig, AlgoIMPALAConfig, Dict[str, Any]]

RLLIB_ALGO_CONFIG_MAP = {
    "ppo": AlgoPPOConfig,
    "impala": AlgoIMPALAConfig,
}


@dataclass
class AlgorithmConfig:
    name: str
    iterations: int
    config: Optional[AlgoConfigType] = None

    @classmethod
    def from_dict(cls, data: dict, backend: str = "rllib") -> "AlgorithmConfig":
        name = data.get("name")
        iterations = data.get("iterations")
        config_data = data.get("config", {}) or {}

        if name is None:
            raise ValueError("Algorithm name is required")
        if iterations is None:
            raise ValueError("Algorithm iterations is required")

        if backend == "rllib":
            config_class = RLLIB_ALGO_CONFIG_MAP.get(name)
            if config_class is None:
                raise ValueError(f"Unknown RLlib algorithm: {name}. Supported: {list(RLLIB_ALGO_CONFIG_MAP.keys())}")

            policy_data = config_data.pop("policy", None)
            policy = PolicyConfig.from_dict(policy_data) if policy_data else None

            if "policy" in config_class.__dataclass_fields__:
                config = config_class(**config_data, policy=policy)
            else:
                config = config_class(**config_data)
        else:
            config = config_data

        return cls(name=name, iterations=iterations, config=config)
