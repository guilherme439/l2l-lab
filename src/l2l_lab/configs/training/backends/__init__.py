from typing import Annotated, Union

from pydantic import Field

from .alphazoo import AlphazooBackendConfig
from .base import BaseBackendConfig
from .rllib import RllibBackendConfig

BackendConfig = Annotated[
    Union[RllibBackendConfig, AlphazooBackendConfig],
    Field(discriminator="name"),
]

__all__ = [
    "BaseBackendConfig",
    "RllibBackendConfig",
    "AlphazooBackendConfig",
    "BackendConfig",
]
