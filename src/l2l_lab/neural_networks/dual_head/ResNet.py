from typing import Optional

import hexagdly
from alphazoo import AlphaZooNet

from .modules.blocks import *
from .modules.value_heads import *
from .modules.policy_heads import *


class ResNet(AlphaZooNet):

    def __init__(
        self,
        in_channels,
        num_actions,
        num_filters=256,
        num_blocks=4,
        batch_norm=False,
        policy_head="conv-projection",
        value_head="conv-projection",
        policy_channels: Optional[int] = None,
        value_activation="tanh",
        hex=False,
    ):
        super().__init__()

        # Input Module
        input_layers = []
        if hex:
            input_layers.append(hexagdly.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=num_filters, stride=1, bias=False))
        else:
            input_layers.append(nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=num_filters, stride=1, padding='same', bias=False))
        if batch_norm:
            input_layers.append(nn.BatchNorm2d(num_features=num_filters))
        input_layers.append(nn.ReLU())

        self.input_block = nn.Sequential(*input_layers)


        # Processing module
        residual_blocks_list = []
        for b in range(num_blocks):
            residual_blocks_list.append(BasicBlock(num_filters, batch_norm=batch_norm, hex=hex))

        self.residual_blocks = nn.Sequential(*residual_blocks_list)


        # Output Module
        ## POLICY HEAD
        match policy_head:
            case "conv-reduce":
                self.policy_head = ConvReduce_PolicyHead(num_filters, policy_channels, batch_norm=batch_norm, hex=hex)
            case "conv-projection":
                self.policy_head = ConvProjection_PolicyHead(num_filters, num_actions, batch_norm=batch_norm, hex=hex)
            case _:
                raise ValueError(f"Unknown policy_head: {policy_head}")

        ## VALUE HEAD
        match value_head:
            case "conv-reduce":
                self.value_head = ConvReduce_ValueHead(num_filters, activation=value_activation, batch_norm=batch_norm, hex=hex)
            case "conv-projection":
                self.value_head = ConvProjection_ValueHead(num_filters, batch_norm=batch_norm, hex=hex)
            case _:
                raise ValueError(f"Unknown value_head: {value_head}")
    


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
    


    