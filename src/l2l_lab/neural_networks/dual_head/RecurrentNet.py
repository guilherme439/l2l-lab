"""
Adapted from the Deepthinking repository.
"""
from __future__ import annotations

import hexagdly
import torch
from alphazoo import AlphaZooRecurrentNet
from torch import nn

from l2l_lab.configs.training.network import RecurrentNetConfig
from l2l_lab.neural_networks.utils.builders import (build_policy_head,
                                                     build_value_head)

from .modules.blocks import BasicBlock


class RecurrentNet(AlphaZooRecurrentNet):

    def __init__(self, cfg: RecurrentNetConfig, in_channels: int, num_actions: int) -> None:
        super().__init__()

        self.recall = cfg.recall
        self.num_filters = int(cfg.num_filters)

        # all conv layers use bias=False to stay consistent with the DeepThinking architecture.
        if cfg.hex:
            proj_conv = hexagdly.Conv2d(
                in_channels,
                self.num_filters,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            proj_conv = nn.Conv2d(
                in_channels,
                self.num_filters,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=False
            )

        if cfg.hex:
            conv_recall = hexagdly.Conv2d(
                self.num_filters + in_channels,
                self.num_filters,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            conv_recall = nn.Conv2d(
                self.num_filters + in_channels,
                self.num_filters,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=False
            )

        recur_layers: list[nn.Module] = []
        if cfg.recall:
            recur_layers.append(conv_recall)
        for _ in range(cfg.num_blocks):
            recur_layers.append(BasicBlock(self.num_filters, hex=cfg.hex, use_bias=False))

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_module = nn.Sequential(*recur_layers)

        self.policy_head = build_policy_head(
            cfg.policy_head,
            num_filters=self.num_filters,
            num_actions=num_actions
        )
        self.value_head = build_value_head(
            cfg.value_head,
            num_filters=self.num_filters
        )

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        for _ in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_module(interim_thought)

        policy_out = self.policy_head(interim_thought)
        value_out = self.value_head(interim_thought)
        out = (policy_out, value_out)

        return out, interim_thought
