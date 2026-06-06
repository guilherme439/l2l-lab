from dataclasses import dataclass
from typing import Literal

from .base import BasePolicyHeadConfig


@dataclass
class LinearReducePolicyHeadConfig(BasePolicyHeadConfig):
    name: Literal["linear_reduce"] = "linear_reduce"
    num_layers: int = 3
