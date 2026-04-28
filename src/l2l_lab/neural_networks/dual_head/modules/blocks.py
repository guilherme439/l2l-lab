import math

import hexagdly
import torch
import torch.nn.functional as F
from torch import nn

from l2l_lab.neural_networks.utils.activations import make_activation


class BasicBlock(nn.Module):

    def __init__(self, channels, batch_norm=False, hex=False):
        super().__init__()

        before_shortcut_layers = []

        if hex:
            before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        else:
            before_shortcut_layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 3, padding='same', bias=False))
        if batch_norm:
            before_shortcut_layers.append(nn.BatchNorm2d(num_features=channels))
        before_shortcut_layers.append(nn.ReLU())

        if hex:
            before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        else:
            before_shortcut_layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 3, padding='same', bias=False))

        self.before_shortcut = nn.Sequential(*before_shortcut_layers)
        self.shortcut = nn.Sequential()



    def forward(self, x):
        out = self.before_shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class HighwayBlock(nn.Module):

    def __init__(self, width, num_layers, activation="silu"):
        super().__init__()

        transform_layers = []
        for _ in range(num_layers):
            transform_layers.append(nn.Linear(in_features=width, out_features=width))
            transform_layers.append(make_activation(activation))

        self.transform_block = nn.Sequential(*transform_layers)
        self.gate_layer = nn.Linear(in_features=width, out_features=width)



    def forward(self, x):
        transformed = self.transform_block(x)
        gate_values = torch.sigmoid(self.gate_layer(x))
        return gate_values * transformed + (1 - gate_values) * x
