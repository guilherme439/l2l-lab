from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..algorithms.base import BaseAlgorithmConfig, algorithm_config_from_dict


@dataclass
class BaseBackendConfig:
    name: str
    algorithm: BaseAlgorithmConfig
    continue_training: bool = False
    continue_from_iteration: Optional[int] = None


def backend_config_from_dict(data: dict[str, Any]) -> BaseBackendConfig:
    from .alphazoo import AlphazooBackendConfig
    from .rllib import RllibBackendConfig

    name = data.get("name")
    if name is None:
        raise ValueError("backend.name is required")

    if name == "rllib":
        return RllibBackendConfig._from_dict(data)
    if name == "alphazoo":
        return AlphazooBackendConfig._from_dict(data)
    raise ValueError(f"Unknown backend {name!r} (expected 'rllib' or 'alphazoo')")
