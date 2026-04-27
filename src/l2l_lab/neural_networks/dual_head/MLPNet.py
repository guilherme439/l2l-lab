from typing import Optional

from alphazoo import AlphaZooNet

from .modules.blocks import *
from .modules.policy_heads import *
from .modules.value_heads import *


class MLPNet(AlphaZooNet):

    def __init__(self, out_features, input_features, hidden_layers=4, neurons_per_layer=64, head_layers=3, highway_interval: Optional[int] = None):

        super().__init__()

        # General Module
        general_module_layers = [
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=neurons_per_layer),
            nn.SiLU(),
        ]
        if highway_interval is None:
            for _ in range(hidden_layers):
                general_module_layers.append(nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer))
                general_module_layers.append(nn.SiLU())
        else:
            num_highway_blocks = hidden_layers // highway_interval
            plain_layers_remainder = hidden_layers % highway_interval
            for _ in range(num_highway_blocks):
                general_module_layers.append(HighwayBlock(neurons_per_layer, highway_interval))
            for _ in range(plain_layers_remainder):
                general_module_layers.append(nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer))
                general_module_layers.append(nn.SiLU())

        self.general_module = nn.Sequential(*general_module_layers)

        

        # Policy Head
        self.policy_head = ReduceMLP_PolicyHead(neurons_per_layer, out_features, num_layers=head_layers)

        # Value Head
        self.value_head = ReduceMLP_ValueHead(neurons_per_layer, num_layers=head_layers)
        


    def forward_trunk(self, x):
        return self.general_module(x)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)
