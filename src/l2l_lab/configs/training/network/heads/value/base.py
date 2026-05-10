from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BaseValueHeadConfig:
    name: str
    activation: str = "relu"
    final_activation: Optional[str] = None

    def validate_for_env(self, state_shape: Tuple[int, ...], num_actions: int) -> None:
        return None
