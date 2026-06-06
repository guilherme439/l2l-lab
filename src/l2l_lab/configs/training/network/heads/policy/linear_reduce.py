from dataclasses import dataclass

from .base import BasePolicyHeadConfig


@dataclass
class LinearReducePolicyHeadConfig(BasePolicyHeadConfig):
    name: str = "linear_reduce"
    num_layers: int = 3
