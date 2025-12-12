from torch import nn

from .modules.blocks import *
from .modules.value_heads import *
from .modules.policy_heads import *

class MLPNet(nn.Module):

    def __init__(self, out_features, hidden_layers=4, neurons_per_layer=64, head_layers=3):

        super(MLPNet, self).__init__()
        self.recurrent=False

        # General Module
        general_module_layers = [nn.Flatten(), nn.LazyLinear(neurons_per_layer), nn.SiLU()] # First layer and activation
        for layer in range(hidden_layers):
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
    



    