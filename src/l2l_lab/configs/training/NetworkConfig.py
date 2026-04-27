from dataclasses import dataclass, field
from typing import Any, Dict, Type

from l2l_lab.neural_networks.dual_head.ConvNet import ConvNet
from l2l_lab.neural_networks.dual_head.ResNet import ResNet
from l2l_lab.neural_networks.dual_head.MLPNet import MLPNet
from l2l_lab.neural_networks.dual_head.RecurrentNet import RecurrentNet
from l2l_lab.neural_networks.dual_head.SNNet import SNNet

CONV_ARCHITECTURES = {"ResNet", "ConvNet", "RecurrentNet"}
MLP_ARCHITECTURES = {"MLPNet", "SNNet"}

@dataclass
class NetworkConfig:
    architecture: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        return self.kwargs.copy()

    def get_network_class(self) -> Type:
        match self.architecture:
            case "ResNet":
                return ResNet
            case "ConvNet":
                return ConvNet
            case "MLPNet":
                return MLPNet
            case "SNNet":
                return SNNet
            case "RecurrentNet":
                return RecurrentNet
            case _:
                raise ValueError(f"Unknown architecture: {self.architecture}")
