from typing import Annotated, Union

from pydantic import Field

from .base import BaseNetworkConfig
from .convnet import ConvNetConfig
from .heads import (BasePolicyHeadConfig, BaseValueHeadConfig,
                    ConvProjectionPolicyHeadConfig,
                    ConvProjectionValueHeadConfig, ConvReducePolicyHeadConfig,
                    ConvReduceValueHeadConfig, LinearReducePolicyHeadConfig,
                    LinearReduceValueHeadConfig, PolicyHeadConfig,
                    ValueHeadConfig)
from .mlpnet import MLPNetConfig
from .recurrentnet import RecurrentNetConfig
from .resnet import ResNetConfig
from .snnet import SNNetConfig

NetworkConfig = Annotated[
    Union[ResNetConfig, ConvNetConfig, RecurrentNetConfig, MLPNetConfig, SNNetConfig],
    Field(discriminator="architecture"),
]

__all__ = [
    "BaseNetworkConfig",
    "NetworkConfig",
    "ResNetConfig",
    "ConvNetConfig",
    "RecurrentNetConfig",
    "MLPNetConfig",
    "SNNetConfig",
    "BasePolicyHeadConfig",
    "ConvProjectionPolicyHeadConfig",
    "ConvReducePolicyHeadConfig",
    "LinearReducePolicyHeadConfig",
    "PolicyHeadConfig",
    "BaseValueHeadConfig",
    "ConvProjectionValueHeadConfig",
    "ConvReduceValueHeadConfig",
    "LinearReduceValueHeadConfig",
    "ValueHeadConfig",
]
