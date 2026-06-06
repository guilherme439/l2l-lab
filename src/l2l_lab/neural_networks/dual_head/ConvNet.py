import hexagdly
from alphazoo import AlphaZooNet
from torch import nn

from l2l_lab.configs.training.network import ConvNetConfig
from l2l_lab.neural_networks.utils.builders import (build_policy_head,
                                                     build_value_head)


class ConvNet(AlphaZooNet):

    def __init__(self, cfg: ConvNetConfig, in_channels: int, num_actions: int) -> None:
        super().__init__()
        self.num_filters = cfg.num_filters

        general_layer_list: list[nn.Module] = []
        if cfg.hex:
            general_layer_list.append(hexagdly.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=self.num_filters
            ))
        else:
            general_layer_list.append(nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=self.num_filters,
                padding='same'
            ))
        general_layer_list.append(nn.ELU())

        for _ in range(cfg.num_layers):
            if cfg.hex:
                general_layer_list.append(hexagdly.Conv2d(
                    kernel_size=1,
                    in_channels=self.num_filters,
                    out_channels=self.num_filters
                ))
            else:
                general_layer_list.append(nn.Conv2d(
                    kernel_size=3,
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    padding='same'
                ))
            general_layer_list.append(nn.ELU())

        self.general_module = nn.Sequential(*general_layer_list)

        self.policy_head = build_policy_head(
            cfg.policy_head,
            num_filters=self.num_filters,
            num_actions=num_actions
        )
        self.value_head = build_value_head(
            cfg.value_head,
            num_filters=self.num_filters
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
