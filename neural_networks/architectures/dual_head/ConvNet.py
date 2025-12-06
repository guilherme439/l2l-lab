import hexagdly
from torch import nn

from .modules.blocks import *
from .modules.value_heads import *
from .modules.policy_heads import *


class ConvNet(nn.Module):

    def __init__(self, in_channels, policy_channels, num_filters=256, num_layers=6, hex=True):

        super().__init__()
        self.recurrent = False
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
        self.policy_head = Reduce_PolicyHead(self.num_filters, policy_channels, hex=hex)
        

        # Value Head
        self.value_head = Reduce_ValueHead(self.num_filters, hex=hex)


    def forward_trunk(self, x):
        return self.general_module(x)

    def forward_heads(self, embeddings):
        policy = self.policy_head(embeddings)
        value = self.value_head(embeddings)
        return policy, value

    def forward(self, x):
        embeddings = self.forward_trunk(x)
        return self.forward_heads(embeddings)
    



    