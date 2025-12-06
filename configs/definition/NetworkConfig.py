from dataclasses import dataclass, field
from typing import Any, Dict, Type

from neural_networks.architectures.dual_head.ConvNet import ConvNet
from neural_networks.architectures.dual_head.ResNet import ResNet
from neural_networks.architectures.dual_head.MLPNet import MLPNet

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
            case _:
                raise ValueError(f"Unknown architecture: {self.architecture}")
