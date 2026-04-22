from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..algorithms.base import (AlphazooAlgorithmConfig,
                               algorithm_config_from_dict)
from .base import BaseBackendConfig


@dataclass
class AlphazooBackendConfig(BaseBackendConfig):
    load_scheduler: Optional[bool] = None
    load_optimizer: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.name != "alphazoo":
            raise ValueError(
                f"AlphazooBackendConfig requires name='alphazoo', got {self.name!r}"
            )
        if not isinstance(self.algorithm, AlphazooAlgorithmConfig):
            raise ValueError(
                "alphazoo backend requires an AlphazooAlgorithmConfig, "
                f"got {type(self.algorithm).__name__}"
            )

        if self.load_scheduler is None:
            self.load_scheduler = self.continue_training
        if self.load_optimizer is None:
            self.load_optimizer = self.continue_training

        if not self.continue_training and (self.load_scheduler or self.load_optimizer):
            raise ValueError(
                "load_scheduler/load_optimizer require continue_training=true; "
                "nothing to load from when starting a fresh run."
            )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "AlphazooBackendConfig":
        algorithm = algorithm_config_from_dict(data.get("algorithm", {}) or {}, "alphazoo")
        return cls(
            name=data["name"],
            algorithm=algorithm,
            continue_training=data.get("continue_training", False),
            continue_from_iteration=data.get("continue_from_iteration"),
            load_scheduler=data.get("load_scheduler"),
            load_optimizer=data.get("load_optimizer"),
        )
