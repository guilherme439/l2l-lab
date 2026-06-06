from dataclasses import dataclass
from typing import Literal

from .base import BaseValueHeadConfig


@dataclass
class LinearReduceValueHeadConfig(BaseValueHeadConfig):
    name: Literal["linear_reduce"] = "linear_reduce"
    activation: str = "tanh"
    num_layers: int = 3
