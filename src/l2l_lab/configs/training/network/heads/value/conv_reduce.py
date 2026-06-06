from dataclasses import dataclass
from typing import Literal, Optional

from .base import BaseValueHeadConfig


@dataclass
class ConvReduceValueHeadConfig(BaseValueHeadConfig):
    name: Literal["conv_reduce"] = "conv_reduce"
    activation: str = "tanh"
    num_reduce_layers: int = 4
    batch_norm: Optional[bool] = None
    hex: Optional[bool] = None
