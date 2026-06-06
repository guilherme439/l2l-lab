from dataclasses import dataclass
from typing import Optional, override

from .base import BasePolicyHeadConfig


@dataclass
class ConvReducePolicyHeadConfig(BasePolicyHeadConfig):
    name: str = "conv_reduce"
    policy_channels: int = 0
    num_reduce_layers: int = 2
    batch_norm: Optional[bool] = None
    hex: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.policy_channels <= 0:
            raise ValueError(
                "ConvReducePolicyHeadConfig requires policy_channels > 0"
            )

    @override
    def validate_for_env(self, state_shape: tuple[int, ...], num_actions: int) -> None:
        h, w = state_shape[1], state_shape[2]
        expected = self.policy_channels * h * w
        if expected != num_actions:
            raise ValueError(
                f"conv_reduce policy head requires policy_channels * H * W == num_actions: "
                f"got {self.policy_channels} * {h} * {w} = {expected} != {num_actions}."
            )
