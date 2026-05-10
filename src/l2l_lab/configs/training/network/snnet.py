from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import BaseNetworkConfig


@dataclass
class SNNetConfig(BaseNetworkConfig):
    architecture: str = "SNNet"
    hidden_layers: int = 6
    neurons_per_layer: int = 256
    head_layers: int = 2
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.architecture != "SNNet":
            raise ValueError(
                f"SNNetConfig requires architecture='SNNet', got {self.architecture!r}"
            )

    def is_recurrent(self) -> bool:
        return False

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "SNNetConfig":
        kwargs: Dict[str, Any] = {
            "architecture": data.get("architecture", "SNNet")
        }
        for key in ("hidden_layers", "neurons_per_layer", "head_layers", "dropout"):
            if key in data:
                kwargs[key] = data[key]
        return cls(**kwargs)
