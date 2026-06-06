from dataclasses import dataclass
from typing import Literal, Optional

from .base import BaseValueHeadConfig


@dataclass
class ConvProjectionValueHeadConfig(BaseValueHeadConfig):
    name: Literal["conv_projection"] = "conv_projection"
    dense_layer_neurons: int = 256
    conv_layer_channels: int = 32
    batch_norm: Optional[bool] = None
    hex: Optional[bool] = None
