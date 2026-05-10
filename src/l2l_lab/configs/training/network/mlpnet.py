from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import BaseNetworkConfig
from .heads import (LinearReducePolicyHeadConfig, LinearReduceValueHeadConfig,
                    policy_head_from_dict, value_head_from_dict)


@dataclass
class MLPNetConfig(BaseNetworkConfig):
    architecture: str = "MLPNet"
    hidden_layers: int = 4
    neurons_per_layer: int = 64
    highway_interval: Optional[int] = None
    policy_head: LinearReducePolicyHeadConfig = field(
        default_factory=lambda: LinearReducePolicyHeadConfig()
    )
    value_head: LinearReduceValueHeadConfig = field(
        default_factory=lambda: LinearReduceValueHeadConfig()
    )

    def __post_init__(self) -> None:
        if self.architecture != "MLPNet":
            raise ValueError(
                f"MLPNetConfig requires architecture='MLPNet', got {self.architecture!r}"
            )
        if not isinstance(self.policy_head, LinearReducePolicyHeadConfig):
            raise ValueError(
                f"MLPNet only supports linear_reduce policy heads, "
                f"got {type(self.policy_head).__name__}"
            )
        if not isinstance(self.value_head, LinearReduceValueHeadConfig):
            raise ValueError(
                f"MLPNet only supports linear_reduce value heads, "
                f"got {type(self.value_head).__name__}"
            )

    def is_recurrent(self) -> bool:
        return False

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "MLPNetConfig":
        policy_head_data = data.get("policy_head")
        value_head_data = data.get("value_head")
        kwargs: Dict[str, Any] = {
            "architecture": data.get("architecture", "MLPNet"),
        }
        for key in ("hidden_layers", "neurons_per_layer", "highway_interval"):
            if key in data:
                kwargs[key] = data[key]
        if policy_head_data is not None:
            kwargs["policy_head"] = policy_head_from_dict(policy_head_data)
        if value_head_data is not None:
            kwargs["value_head"] = value_head_from_dict(value_head_data)
        return cls(**kwargs)
