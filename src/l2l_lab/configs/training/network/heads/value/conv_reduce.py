from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseValueHeadConfig


@dataclass
class ConvReduceValueHeadConfig(BaseValueHeadConfig):
    name: str = "conv_reduce"
    activation: str = "tanh"
    num_reduce_layers: int = 4
    batch_norm: Optional[bool] = None
    hex: Optional[bool] = None
