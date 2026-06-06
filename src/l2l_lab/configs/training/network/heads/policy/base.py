from dataclasses import dataclass
from typing import Optional


@dataclass
class BasePolicyHeadConfig:
    name: str
    activation: str = "relu"
    final_activation: Optional[str] = None

    def validate_for_env(self, state_shape: tuple[int, ...], num_actions: int) -> None:
        return None
