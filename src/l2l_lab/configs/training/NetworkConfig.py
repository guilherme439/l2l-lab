from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type

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

    def validate_for_env(self, state_shape: Tuple[int, ...], num_actions: int) -> None:
        """Raises ValueError if the network kwargs are incompatible with the env shapes.

        For conv-based architectures, ensures policy_head='conv-reduce' is paired with a
        policy_channels value satisfying policy_channels * H * W == num_actions.
        """
        if self.architecture not in CONV_ARCHITECTURES:
            return
        policy_head = self.kwargs.get("policy_head", "conv-projection")
        if policy_head == "conv-reduce":
            policy_channels = self.kwargs.get("policy_channels")
            if policy_channels is None:
                raise ValueError(
                    "network.policy_head='conv-reduce' requires 'policy_channels' to be set in network config."
                )
            h, w = state_shape[1], state_shape[2]
            expected = policy_channels * h * w
            if expected != num_actions:
                raise ValueError(
                    f"network.policy_head='conv-reduce' requires policy_channels * H * W == num_actions: "
                    f"got {policy_channels} * {h} * {w} = {expected} != {num_actions}."
                )

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
