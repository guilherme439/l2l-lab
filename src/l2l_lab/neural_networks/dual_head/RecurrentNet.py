"""
Adapted from the Deepthinking repository.
"""

from typing import Optional

import hexagdly
import torch
from alphazoo import AlphaZooRecurrentNet

from .modules.blocks import *
from .modules.policy_heads import *
from .modules.value_heads import *

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class RecurrentNet(AlphaZooRecurrentNet):

    def __init__(
        self,
        in_channels,
        num_actions,
        num_filters=256,
        num_blocks=2,
        recall=True,
        policy_head="conv-projection",
        value_head="conv-projection",
        policy_channels: Optional[int] = None,
        value_activation="tanh",
        hex=False,
    ):
        super().__init__()
        
        self.recall = recall
        self.num_filters = int(num_filters)
        if hex:
            proj_conv = hexagdly.Conv2d(in_channels, num_filters, kernel_size=1, stride=1, bias=False)
        else:
            proj_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding='same', bias=False)

        if hex:
            conv_recall = hexagdly.Conv2d(num_filters + in_channels, num_filters, kernel_size=1, stride=1, bias=False)
        else:
            conv_recall = nn.Conv2d(num_filters + in_channels, num_filters, kernel_size=3, stride=1, padding='same', bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for b in range(num_blocks):
            recur_layers.append(BasicBlock(self.num_filters, hex=hex))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_module = nn.Sequential(*recur_layers)


        ## POLICY HEAD
        match policy_head:
            case "conv-reduce":
                self.policy_head = ConvReduce_PolicyHead(num_filters, policy_channels, hex=hex)
            case "conv-projection":
                self.policy_head = ConvProjection_PolicyHead(num_filters, num_actions, hex=hex)
            case _:
                raise ValueError(f"Unknown policy_head: {policy_head}")


        ## VALUE HEAD
        match value_head:
            case "conv-reduce":
                self.value_head = ConvReduce_ValueHead(num_filters, activation=value_activation, hex=hex)
            case "conv-projection":
                self.value_head = ConvProjection_ValueHead(num_filters, hex=hex)
            case _:
                raise ValueError(f"Unknown value_head: {value_head}")

        


    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought


        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_module(interim_thought)
        

        policy_out = self.policy_head(interim_thought)
        value_out = self.value_head(interim_thought)
        out = (policy_out, value_out)
        
        return out, interim_thought




