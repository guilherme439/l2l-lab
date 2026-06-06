from dataclasses import dataclass, field
from typing import Literal, Optional, override

from .base import BaseNetworkConfig
from .heads import LinearReducePolicyHeadConfig, LinearReduceValueHeadConfig


@dataclass
class MLPNetConfig(BaseNetworkConfig):
    architecture: Literal["MLPNet"] = "MLPNet"
    hidden_layers: int = 4
    neurons_per_layer: int = 64
    highway_interval: Optional[int] = None
    policy_head: LinearReducePolicyHeadConfig = field(
        default_factory=LinearReducePolicyHeadConfig
    )
    value_head: LinearReduceValueHeadConfig = field(
        default_factory=LinearReduceValueHeadConfig
    )

    @override
    def is_recurrent(self) -> bool:
        return False
