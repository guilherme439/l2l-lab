from dataclasses import dataclass
from typing import Literal, override

from .base import BaseNetworkConfig


@dataclass
class SNNetConfig(BaseNetworkConfig):
    architecture: Literal["SNNet"] = "SNNet"
    hidden_layers: int = 6
    neurons_per_layer: int = 256
    head_layers: int = 2
    dropout: float = 0.05

    @override
    def is_recurrent(self) -> bool:
        return False
