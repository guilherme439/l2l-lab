from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from alphazoo import AlphaZooConfig

from ...utils import dataclass_from_dict
from .algo_impala_config import AlgoIMPALAConfig
from .algo_ppo_config import AlgoPPOConfig

_RLLIB_INNER_BY_NAME: dict[str, type] = {
    "ppo": AlgoPPOConfig,
    "impala": AlgoIMPALAConfig,
}


@dataclass
class BaseAlgorithmConfig:
    name: str

    @property
    def total_iterations(self) -> int:
        raise NotImplementedError


@dataclass
class RllibAlgorithmConfig(BaseAlgorithmConfig):
    iterations: int = 0
    config: Union[AlgoPPOConfig, AlgoIMPALAConfig] = field(default_factory=AlgoPPOConfig)

    def __post_init__(self) -> None:
        expected = _RLLIB_INNER_BY_NAME.get(self.name)
        if expected is None:
            raise ValueError(
                f"{self.name!r} is not a supported rllib algorithm "
                f"(expected one of {sorted(_RLLIB_INNER_BY_NAME)})"
            )
        if not isinstance(self.config, expected):
            raise ValueError(
                f"algorithm {self.name!r} requires a {expected.__name__} "
                f"config, got {type(self.config).__name__}"
            )
        if self.iterations <= 0:
            raise ValueError("rllib algorithm needs iterations > 0")

    @property
    def total_iterations(self) -> int:
        return self.iterations


@dataclass
class AlphazooAlgorithmConfig(BaseAlgorithmConfig):
    config: AlphaZooConfig = field(default_factory=AlphaZooConfig)

    def __post_init__(self) -> None:
        if self.name != "alphazero":
            raise ValueError(
                f"{self.name!r} is not a supported alphazoo algorithm "
                f"(expected 'alphazero')"
            )

    @property
    def total_iterations(self) -> int:
        return self.config.running.training_steps


def algorithm_config_from_dict(data: dict[str, Any], backend_name: str) -> BaseAlgorithmConfig:
    name = data.get("name")
    if name is None:
        raise ValueError("algorithm.name is required")

    if backend_name == "rllib":
        inner_cls = _RLLIB_INNER_BY_NAME.get(name)
        if inner_cls is None:
            raise ValueError(
                f"{name!r} is not a rllib algorithm "
                f"(expected one of {sorted(_RLLIB_INNER_BY_NAME)})"
            )
        iterations = data.get("iterations")
        if iterations is None:
            raise ValueError("rllib algorithm requires an 'iterations' field")
        inner = dataclass_from_dict(inner_cls, data.get("config", {}) or {})
        return RllibAlgorithmConfig(name=name, iterations=iterations, config=inner)

    if backend_name == "alphazoo":
        if name != "alphazero":
            raise ValueError(f"{name!r} is not an alphazoo algorithm (expected 'alphazero')")
        inner = dataclass_from_dict(AlphaZooConfig, data.get("config", {}) or {})
        return AlphazooAlgorithmConfig(name=name, config=inner)

    raise ValueError(f"Unknown backend {backend_name!r}")
