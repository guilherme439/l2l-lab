from dataclasses import dataclass
from typing import Optional

from .base import BasePolicyHeadConfig


@dataclass
class ConvProjectionPolicyHeadConfig(BasePolicyHeadConfig):
    name: str = "conv_projection"
    dense_layer_neurons: int = 256
    conv_layer_channels: int = 32
    batch_norm: Optional[bool] = None
    hex: Optional[bool] = None
