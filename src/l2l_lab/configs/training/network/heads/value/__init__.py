from typing import Annotated, Union

from pydantic import Field

from .base import BaseValueHeadConfig
from .conv_projection import ConvProjectionValueHeadConfig
from .conv_reduce import ConvReduceValueHeadConfig
from .linear_reduce import LinearReduceValueHeadConfig

ValueHeadConfig = Annotated[
    Union[
        ConvProjectionValueHeadConfig,
        ConvReduceValueHeadConfig,
        LinearReduceValueHeadConfig,
    ],
    Field(discriminator="name"),
]

__all__ = [
    "BaseValueHeadConfig",
    "ConvProjectionValueHeadConfig",
    "ConvReduceValueHeadConfig",
    "LinearReduceValueHeadConfig",
    "ValueHeadConfig",
]
