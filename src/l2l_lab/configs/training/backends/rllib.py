from dataclasses import dataclass
from typing import Literal

from ..algorithms.base import RllibAlgorithmConfig
from .base import BaseBackendConfig


@dataclass
class RllibBackendConfig(BaseBackendConfig):
    name: Literal["rllib"]

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, RllibAlgorithmConfig):
            raise ValueError(
                "rllib backend requires a RllibAlgorithmConfig, "
                f"got {type(self.algorithm).__name__}"
            )
