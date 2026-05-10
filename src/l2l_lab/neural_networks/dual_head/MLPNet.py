from __future__ import annotations

from alphazoo import AlphaZooNet
from torch import nn

from l2l_lab.configs.training.network import MLPNetConfig
from l2l_lab.neural_networks.utils.builders import (build_policy_head,
                                                     build_value_head)

from .modules.blocks import HighwayBlock


class MLPNet(AlphaZooNet):

    def __init__(self, cfg: MLPNetConfig, input_features: int, num_actions: int) -> None:
        super().__init__()

        general_module_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=cfg.neurons_per_layer),
            nn.SiLU(),
        ]
        if cfg.highway_interval is None:
            for _ in range(cfg.hidden_layers):
                general_module_layers.append(nn.Linear(
                    in_features=cfg.neurons_per_layer, out_features=cfg.neurons_per_layer,
                ))
                general_module_layers.append(nn.SiLU())
        else:
            num_highway_blocks = cfg.hidden_layers // cfg.highway_interval
            plain_layers_remainder = cfg.hidden_layers % cfg.highway_interval
            for _ in range(num_highway_blocks):
                general_module_layers.append(HighwayBlock(cfg.neurons_per_layer, cfg.highway_interval))
            for _ in range(plain_layers_remainder):
                general_module_layers.append(nn.Linear(
                    in_features=cfg.neurons_per_layer, out_features=cfg.neurons_per_layer,
                ))
                general_module_layers.append(nn.SiLU())

        self.general_module = nn.Sequential(*general_module_layers)

        self.policy_head = build_policy_head(
            cfg.policy_head,
            in_features=cfg.neurons_per_layer,
            out_features=num_actions,
        )
        self.value_head = build_value_head(
            cfg.value_head,
            in_features=cfg.neurons_per_layer,
        )

    def forward_trunk(self, x):
        return self.general_module(x)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)
