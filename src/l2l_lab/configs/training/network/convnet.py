from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .base import BaseNetworkConfig
from .heads import (BasePolicyHeadConfig, BaseValueHeadConfig,
                    ConvProjectionPolicyHeadConfig,
                    ConvProjectionValueHeadConfig,
                    LinearReducePolicyHeadConfig,
                    LinearReduceValueHeadConfig, policy_head_from_dict,
                    value_head_from_dict)


@dataclass
class ConvNetConfig(BaseNetworkConfig):
    architecture: str = "ConvNet"
    num_filters: int = 256
    num_layers: int = 6
    hex: bool = False
    policy_head: BasePolicyHeadConfig = field(
        default_factory=lambda: ConvProjectionPolicyHeadConfig()
    )
    value_head: BaseValueHeadConfig = field(
        default_factory=lambda: ConvProjectionValueHeadConfig()
    )

    def __post_init__(self) -> None:
        if self.architecture != "ConvNet":
            raise ValueError(
                f"ConvNetConfig requires architecture='ConvNet', got {self.architecture!r}"
            )
        if isinstance(self.policy_head, LinearReducePolicyHeadConfig):
            raise ValueError("ConvNet does not support linear_reduce policy heads")
        if isinstance(self.value_head, LinearReduceValueHeadConfig):
            raise ValueError("ConvNet does not support linear_reduce value heads")

        for head in (self.policy_head, self.value_head):
            if head.batch_norm is None:
                head.batch_norm = False
            if head.hex is None:
                head.hex = self.hex

    def is_recurrent(self) -> bool:
        return False

    def validate_for_env(self, state_shape: Tuple[int, ...], num_actions: int) -> None:
        self.policy_head.validate_for_env(state_shape, num_actions)
        self.value_head.validate_for_env(state_shape, num_actions)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "ConvNetConfig":
        policy_head_data = data.get("policy_head")
        value_head_data = data.get("value_head")
        kwargs: Dict[str, Any] = {
            "architecture": data.get("architecture", "ConvNet"),
        }
        for key in ("num_filters", "num_layers", "hex"):
            if key in data:
                kwargs[key] = data[key]
        if policy_head_data is not None:
            kwargs["policy_head"] = policy_head_from_dict(policy_head_data)
        if value_head_data is not None:
            kwargs["value_head"] = value_head_from_dict(value_head_data)
        return cls(**kwargs)
