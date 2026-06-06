from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter


@dataclass
class BaseNetworkConfig:
    architecture: str

    def is_recurrent(self) -> bool:
        raise NotImplementedError

    def validate_for_env(self, state_shape: tuple[int, ...], num_actions: int) -> None:
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseNetworkConfig:
        from l2l_lab.configs.training.network import NetworkConfig
        return TypeAdapter(NetworkConfig).validate_python(data)
