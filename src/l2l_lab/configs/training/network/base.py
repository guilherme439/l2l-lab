from dataclasses import dataclass
from typing import Any


@dataclass
class BaseNetworkConfig:
    architecture: str

    def is_recurrent(self) -> bool:
        raise NotImplementedError

    def validate_for_env(self, state_shape: tuple[int, ...], num_actions: int) -> None:
        return None


def network_config_from_dict(data: dict[str, Any]) -> BaseNetworkConfig:
    architecture = data.get("architecture")
    if architecture is None:
        raise ValueError("network config requires an 'architecture' field")

    from .convnet import ConvNetConfig
    from .mlpnet import MLPNetConfig
    from .recurrentnet import RecurrentNetConfig
    from .resnet import ResNetConfig
    from .snnet import SNNetConfig

    match architecture:
        case "ResNet":
            return ResNetConfig._from_dict(data)
        case "ConvNet":
            return ConvNetConfig._from_dict(data)
        case "RecurrentNet":
            return RecurrentNetConfig._from_dict(data)
        case "MLPNet":
            return MLPNetConfig._from_dict(data)
        case "SNNet":
            return SNNetConfig._from_dict(data)
        case _:
            raise ValueError(
                f"Unknown architecture {architecture!r} "
                f"(expected one of 'ResNet', 'ConvNet', 'RecurrentNet', 'MLPNet', 'SNNet')"
            )
