import hexagdly
from alphazoo import AlphaZooNet
from torch import nn

from l2l_lab.configs.training.network import ResNetConfig
from l2l_lab.neural_networks.utils.builders import (build_policy_head,
                                                     build_value_head)

from .modules.blocks import BasicBlock


class ResNet(AlphaZooNet):

    def __init__(self, cfg: ResNetConfig, in_channels: int, num_actions: int) -> None:
        super().__init__()

        input_layers: list[nn.Module] = []
        if cfg.hex:
            input_layers.append(hexagdly.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=cfg.num_filters,
                stride=1,
                bias=not cfg.batch_norm
            ))
        else:
            input_layers.append(nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=cfg.num_filters,
                stride=1,
                padding='same',
                bias=not cfg.batch_norm
            ))
        if cfg.batch_norm:
            input_layers.append(nn.BatchNorm2d(num_features=cfg.num_filters))
        input_layers.append(nn.ReLU())
        self.input_block = nn.Sequential(*input_layers)

        self.residual_blocks = nn.Sequential(*[
            BasicBlock(cfg.num_filters, batch_norm=cfg.batch_norm, hex=cfg.hex)
            for _ in range(cfg.num_blocks)
        ])

        self.policy_head = build_policy_head(
            cfg.policy_head,
            num_filters=cfg.num_filters,
            num_actions=num_actions
        )
        self.value_head = build_value_head(
            cfg.value_head,
            num_filters=cfg.num_filters
        )

    def forward_trunk(self, x):
        projection = self.input_block(x)
        return self.residual_blocks(projection)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)
