"""Self-Normalizing MLP architecture (Klambauer et al. 2017).

The trunk uses SELU activations together with LeCun-normal weight init so
activation statistics stay at the (mean=0, var=1) fixed point through depth
without normalization layers or skip connections.
"""

import math

from alphazoo import AlphaZooNet
from torch import nn

from .modules.policy_heads import ReduceMLP_PolicyHead
from .modules.value_heads import ReduceMLP_ValueHead


class SNNet(AlphaZooNet):

    def __init__(
        self,
        out_features: int,
        input_features: int,
        hidden_layers: int = 6,
        neurons_per_layer: int = 256,
        head_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        trunk_layers: list[nn.Module] = [
            nn.Flatten(),
            # centers/rescales raw observations so the trunk sees the (mean=0, var=1)
            # input distribution that SELU's fixed-point math assumes.
            nn.LayerNorm(input_features, elementwise_affine=False),
            nn.Linear(in_features=input_features, out_features=neurons_per_layer),
            nn.SELU(),
        ]
        # AlphaDropout preserves SELU's mean=0/var=1 fixed point; nn.Dropout would break it.
        if dropout > 0:
            trunk_layers.append(nn.AlphaDropout(p=dropout))
        for _ in range(hidden_layers):
            trunk_layers.append(nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer))
            trunk_layers.append(nn.SELU())
            if dropout > 0:
                trunk_layers.append(nn.AlphaDropout(p=dropout))
        self.general_module = nn.Sequential(*trunk_layers)

        self.policy_head = ReduceMLP_PolicyHead(
            neurons_per_layer, out_features, num_layers=head_layers, activation="selu",
        )
        self.value_head = ReduceMLP_ValueHead(
            neurons_per_layer, num_layers=head_layers, activation="selu",
        )

        self.apply(_lecun_normal_init)

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)

    def forward_trunk(self, x):
        return self.general_module(x)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value


def _lecun_normal_init(module: nn.Module) -> None:
    # SELU's self-normalizing fixed point only holds when weights come from
    # Normal(0, 1/sqrt(fan_in)); deviating from this scale breaks variance
    # preservation across depth. PyTorch's default Linear init uses a different
    # scheme (Kaiming-uniform with a≈sqrt(5)) which is fine for ReLU/SiLU but
    # wrong for SELU.
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
