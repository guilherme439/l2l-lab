
import hexagdly

import torch.nn.functional as F
from torch import nn



class Reduce_PolicyHead(nn.Module):
    '''Several conv layers that progressively reduce the amount of filters,
       until the policy's number of channels is reached'''

    def __init__(self, width, policy_channels, num_reduce_layers=2, batch_norm=False, hex=True):
        super().__init__()

        layer_list = []

        delta = policy_channels - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers, 0, -1):
            current_layer_filters = previous_layer_filters + step
            if hex:
                layer_list.append(hexagdly.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=1,
                    stride=1,
                    bias=False
                ))
            else:
                layer_list.append(nn.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    bias=False
                ))

            if layer != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))

                layer_list.append(nn.ReLU())

            previous_layer_filters = current_layer_filters


        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    

##################################################################################################


class ReduceMLP_PolicyHead(nn.Module):

    def __init__(self, in_features, out_features, num_layers=3):
        super().__init__()

        layer_list = []

        delta = out_features - in_features
        step = delta / num_layers
        previous_layer_features = in_features

        for layer in range(num_layers, 0, -1):
            current_layer_features = previous_layer_features + step

            layer_list.append(nn.Linear(
                max(1, int(previous_layer_features)),
                max(1, int(current_layer_features)),
            ))

            if layer != 1:
                layer_list.append(nn.ReLU())

            previous_layer_features = current_layer_features

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)