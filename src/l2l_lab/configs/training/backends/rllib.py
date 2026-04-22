from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..algorithms.base import (RllibAlgorithmConfig,
                               algorithm_config_from_dict)
from .base import BaseBackendConfig


@dataclass
class RllibBackendConfig(BaseBackendConfig):
    def __post_init__(self) -> None:
        if self.name != "rllib":
            raise ValueError(f"RllibBackendConfig requires name='rllib', got {self.name!r}")
        if not isinstance(self.algorithm, RllibAlgorithmConfig):
            raise ValueError(
                "rllib backend requires a RllibAlgorithmConfig, "
                f"got {type(self.algorithm).__name__}"
            )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "RllibBackendConfig":
        algorithm = algorithm_config_from_dict(data.get("algorithm", {}) or {}, "rllib")
        return cls(
            name=data["name"],
            algorithm=algorithm,
            continue_training=data.get("continue_training", False),
            continue_from_iteration=data.get("continue_from_iteration"),
        )
