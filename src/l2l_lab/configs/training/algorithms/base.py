from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from alphazoo import AlphaZooConfig
from pydantic import Field

from .algo_impala_config import AlgoIMPALAConfig
from .algo_ppo_config import AlgoPPOConfig


@dataclass
class BaseAlgorithmConfig:
    name: str

    @property
    def total_iterations(self) -> int:
        raise NotImplementedError


@dataclass
class RllibAlgorithmConfig(BaseAlgorithmConfig):
    iterations: int = 0

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("rllib algorithm needs iterations > 0")

    @property
    def total_iterations(self) -> int:
        return self.iterations


@dataclass
class PPOAlgorithmConfig(RllibAlgorithmConfig):
    name: Literal["ppo"] = "ppo"
    config: AlgoPPOConfig = field(default_factory=AlgoPPOConfig)


@dataclass
class IMPALAAlgorithmConfig(RllibAlgorithmConfig):
    name: Literal["impala"] = "impala"
    config: AlgoIMPALAConfig = field(default_factory=AlgoIMPALAConfig)


@dataclass
class AlphazooAlgorithmConfig(BaseAlgorithmConfig):
    name: Literal["alphazero"] = "alphazero"
    config: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, AlphaZooConfig):
            self.config = AlphaZooConfig.from_dict(self.config or {})

    @property
    def total_iterations(self) -> int:
        return self.config.running.training_steps


AlgorithmConfig = Annotated[
    Union[PPOAlgorithmConfig, IMPALAAlgorithmConfig, AlphazooAlgorithmConfig],
    Field(discriminator="name"),
]
