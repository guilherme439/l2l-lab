from dataclasses import dataclass

from .base import BaseValueHeadConfig


@dataclass
class LinearReduceValueHeadConfig(BaseValueHeadConfig):
    name: str = "linear_reduce"
    activation: str = "tanh"
    num_layers: int = 3
