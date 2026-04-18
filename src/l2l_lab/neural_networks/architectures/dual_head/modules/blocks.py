import math

import hexagdly
import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, channels, batch_norm=False, hex=True):
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
    