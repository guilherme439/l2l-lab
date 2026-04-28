from typing import Optional

import hexagdly
from alphazoo import AlphaZooNet

from .modules.blocks import *
from .modules.value_heads import *
from .modules.policy_heads import *


class ConvNet(AlphaZooNet):

    def __init__(
        self,
        in_channels,
        num_actions,
        num_filters=256,
        num_layers=6,
        policy_head="conv-projection",
        value_head="conv-projection",
        policy_channels: Optional[int] = None,
        hex=False,
    ):
        super().__init__()
        self.num_filters = num_filters

        # General Module
        general_layer_list = []
        if hex:
            general_layer_list.append(hexagdly.Conv2d(
                kernel_size=1,
                in_channels=in_channels, 
                out_channels=self.num_filters, 
                bias=False
            ))
        else:
            general_layer_list.append(nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels, 
                out_channels=self.num_filters, 
                padding='same', 
                bias=False
            ))
        general_layer_list.append(nn.ELU())
         
        for i in range(num_layers):
            if hex:
                general_layer_list.append(hexagdly.Conv2d(
                    kernel_size=1,
                    in_channels=self.num_filters, 
                    out_channels=self.num_filters, 
                    bias=False
                ))
            else:
                general_layer_list.append(nn.Conv2d(
                    kernel_size=3,
                    in_channels=self.num_filters, 
                    out_channels=self.num_filters, 
                    padding='same', 
                    bias=False
                ))
            general_layer_list.append(nn.ELU())
        
        
        self.general_module = nn.Sequential(*general_layer_list)
        
    
        # Policy Head
        match policy_head:
            case "conv-reduce":
                self.policy_head = ConvReduce_PolicyHead(self.num_filters, policy_channels, hex=hex)
            case "conv-projection":
                self.policy_head = ConvProjection_PolicyHead(self.num_filters, num_actions, hex=hex)
            case _:
                raise ValueError(f"Unknown policy_head: {policy_head}")

        # Value Head
        match value_head:
            case "conv-reduce":
                self.value_head = ConvReduce_ValueHead(self.num_filters, hex=hex)
            case "conv-projection":
                self.value_head = ConvProjection_ValueHead(self.num_filters, hex=hex)
            case _:
                raise ValueError(f"Unknown value_head: {value_head}")


    def forward_trunk(self, x):
        return self.general_module(x)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)
    



    