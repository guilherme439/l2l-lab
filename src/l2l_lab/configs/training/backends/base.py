from dataclasses import dataclass
from typing import Optional

from ..algorithms.base import AlgorithmConfig


@dataclass
class BaseBackendConfig:
    name: str
    algorithm: AlgorithmConfig
    continue_training: bool = False
    continue_from_iteration: Optional[int] = None
