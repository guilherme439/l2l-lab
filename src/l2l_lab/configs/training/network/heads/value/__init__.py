from __future__ import annotations

from typing import Any

from l2l_lab.configs.utils import dataclass_from_dict

from .base import BaseValueHeadConfig
from .conv_projection import ConvProjectionValueHeadConfig
from .conv_reduce import ConvReduceValueHeadConfig
from .linear_reduce import LinearReduceValueHeadConfig

__all__ = [
    "BaseValueHeadConfig",
    "ConvProjectionValueHeadConfig",
    "ConvReduceValueHeadConfig",
    "LinearReduceValueHeadConfig",
    "value_head_from_dict",
]


def value_head_from_dict(data: dict[str, Any]) -> BaseValueHeadConfig:
    name = data.get("name")
    if name is None:
        raise ValueError("value_head config requires a 'name' field")

    match name:
        case "conv_projection":
            return dataclass_from_dict(ConvProjectionValueHeadConfig, data)
        case "conv_reduce":
            return dataclass_from_dict(ConvReduceValueHeadConfig, data)
        case "linear_reduce":
            return dataclass_from_dict(LinearReduceValueHeadConfig, data)
        case _:
            raise ValueError(
                f"Unknown value_head name {name!r} "
                f"(expected one of 'conv_projection', 'conv_reduce', 'linear_reduce')"
            )
