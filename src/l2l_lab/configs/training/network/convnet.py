from dataclasses import dataclass, field
from typing import Literal, override

from .base import BaseNetworkConfig
from .heads import (ConvProjectionPolicyHeadConfig,
                    ConvProjectionValueHeadConfig, LinearReducePolicyHeadConfig,
                    LinearReduceValueHeadConfig, PolicyHeadConfig,
                    ValueHeadConfig)


@dataclass
class ConvNetConfig(BaseNetworkConfig):
    architecture: Literal["ConvNet"] = "ConvNet"
    num_filters: int = 256
    num_layers: int = 6
    hex: bool = False
    policy_head: PolicyHeadConfig = field(default_factory=ConvProjectionPolicyHeadConfig)
    value_head: ValueHeadConfig = field(default_factory=ConvProjectionValueHeadConfig)

    def __post_init__(self) -> None:
        if isinstance(self.policy_head, LinearReducePolicyHeadConfig):
            raise ValueError("ConvNet does not support linear_reduce policy heads")
        if isinstance(self.value_head, LinearReduceValueHeadConfig):
            raise ValueError("ConvNet does not support linear_reduce value heads")

        for head in (self.policy_head, self.value_head):
            if head.batch_norm is None:
                head.batch_norm = False
            if head.hex is None:
                head.hex = self.hex

    @override
    def is_recurrent(self) -> bool:
        return False

    @override
    def validate_for_env(self, state_shape: tuple[int, ...], num_actions: int) -> None:
        self.policy_head.validate_for_env(state_shape, num_actions)
        self.value_head.validate_for_env(state_shape, num_actions)
