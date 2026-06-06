from typing import Annotated, Union

from pydantic import Field

from .base import BasePolicyHeadConfig
from .conv_projection import ConvProjectionPolicyHeadConfig
from .conv_reduce import ConvReducePolicyHeadConfig
from .linear_reduce import LinearReducePolicyHeadConfig

PolicyHeadConfig = Annotated[
    Union[
        ConvProjectionPolicyHeadConfig,
        ConvReducePolicyHeadConfig,
        LinearReducePolicyHeadConfig,
    ],
    Field(discriminator="name"),
]

__all__ = [
    "BasePolicyHeadConfig",
    "ConvProjectionPolicyHeadConfig",
    "ConvReducePolicyHeadConfig",
    "LinearReducePolicyHeadConfig",
    "PolicyHeadConfig",
]
