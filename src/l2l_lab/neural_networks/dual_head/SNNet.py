"""Self-Normalizing MLP architecture (Klambauer et al. 2017).

The trunk uses SELU activations together with LeCun-normal weight init so
activation statistics stay at the (mean=0, var=1) fixed point through depth
without normalization layers or skip connections.
"""
from __future__ import annotations

import math

from alphazoo import AlphaZooNet
from torch import nn

from l2l_lab.configs.training.network import (LinearReducePolicyHeadConfig,
                                                LinearReduceValueHeadConfig,
                                                SNNetConfig)
from l2l_lab.neural_networks.utils.builders import (build_policy_head,
                                                     build_value_head)


class SNNet(AlphaZooNet):

    def __init__(self, cfg: SNNetConfig, input_features: int, num_actions: int) -> None:
        super().__init__()

        trunk_layers: list[nn.Module] = [
            nn.Flatten(),
            # centers/rescales raw observations so the trunk sees the (mean=0, var=1)
            # input distribution that SELU's fixed-point math assumes.
            nn.LayerNorm(input_features, elementwise_affine=False),
            nn.Linear(in_features=input_features, out_features=cfg.neurons_per_layer),
            nn.SELU()
        ]
        # AlphaDropout preserves SELU's mean=0/var=1 fixed point. nn.Dropout would break it.
        if cfg.dropout > 0:
            trunk_layers.append(nn.AlphaDropout(p=cfg.dropout))
        for _ in range(cfg.hidden_layers):
            trunk_layers.append(nn.Linear(
                in_features=cfg.neurons_per_layer,
                out_features=cfg.neurons_per_layer
            ))
            trunk_layers.append(nn.SELU())
            if cfg.dropout > 0:
                trunk_layers.append(nn.AlphaDropout(p=cfg.dropout))
        self.general_module = nn.Sequential(*trunk_layers)

        policy_head_cfg = LinearReducePolicyHeadConfig(num_layers=cfg.head_layers, activation="selu")
        value_head_cfg = LinearReduceValueHeadConfig(num_layers=cfg.head_layers, activation="selu")

        self.policy_head = build_policy_head(
            policy_head_cfg,
            in_features=cfg.neurons_per_layer,
            out_features=num_actions
        )
        self.value_head = build_value_head(
            value_head_cfg,
            in_features=cfg.neurons_per_layer
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
    # SELU's self-normalizing fixed point only holds when weights come from Normal(0, 1/sqrt(fan_in))
    # Deviating from this scale breaks variance preservation across depth.
    # PyTorch's default Linear init uses a different scheme (Kaiming-uniform with a≈sqrt(5))
    # which is fine for ReLU/SiLU but wrong for SELU.
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
