from __future__ import annotations

from typing import Any

from l2l_lab.configs.utils import dataclass_from_dict

from .base import BasePolicyHeadConfig
from .conv_projection import ConvProjectionPolicyHeadConfig
from .conv_reduce import ConvReducePolicyHeadConfig
from .linear_reduce import LinearReducePolicyHeadConfig

__all__ = [
    "BasePolicyHeadConfig",
    "ConvProjectionPolicyHeadConfig",
    "ConvReducePolicyHeadConfig",
    "LinearReducePolicyHeadConfig",
    "policy_head_from_dict",
]


def policy_head_from_dict(data: dict[str, Any]) -> BasePolicyHeadConfig:
    name = data.get("name")
    if name is None:
        raise ValueError("policy_head config requires a 'name' field")

    match name:
        case "conv_projection":
            return dataclass_from_dict(ConvProjectionPolicyHeadConfig, data)
        case "conv_reduce":
            return dataclass_from_dict(ConvReducePolicyHeadConfig, data)
        case "linear_reduce":
            return dataclass_from_dict(LinearReducePolicyHeadConfig, data)
        case _:
            raise ValueError(
                f"Unknown policy_head name {name!r} "
                f"(expected one of 'conv_projection', 'conv_reduce', 'linear_reduce')"
            )
