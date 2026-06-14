from dataclasses import dataclass
from typing import Literal, Optional

from ..algorithms.base import AlphazooAlgorithmConfig
from .base import BaseBackendConfig


@dataclass
class AlphazooBackendConfig(BaseBackendConfig):
    name: Literal["alphazoo"]
    load_scheduler: Optional[bool] = None
    load_optimizer: Optional[bool] = None
    load_replay_buffer: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, AlphazooAlgorithmConfig):
            raise ValueError(
                "alphazoo backend requires an AlphazooAlgorithmConfig, "
                f"got {type(self.algorithm).__name__}"
            )

        if self.load_scheduler is None:
            self.load_scheduler = self.continue_training
        if self.load_optimizer is None:
            self.load_optimizer = self.continue_training
        if self.load_replay_buffer is None:
            self.load_replay_buffer = self.continue_training

        if not self.continue_training and (self.load_scheduler or self.load_optimizer or self.load_replay_buffer):
            raise ValueError(
                "load_scheduler/load_optimizer/load_replay_buffer require continue_training=true; "
                "nothing to load from when starting a fresh run."
            )
