from __future__ import annotations

from typing import Optional

import hexagdly
from torch import nn

from l2l_lab.neural_networks.utils.builders import build_activation


class ConvReduce_PolicyHead(nn.Module):
    '''Several conv layers that progressively reduce the amount of filters,
       until the policy's number of channels is reached'''

    def __init__(
        self,
        width: int,
        policy_channels: int,
        num_reduce_layers: int = 2,
        activation: str = "relu",
        final_activation: Optional[str] = None,
        batch_norm: bool = False,
        hex: bool = False
    ) -> None:
        super().__init__()

        layer_list: list[nn.Module] = []

        delta = policy_channels - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers, 0, -1):
            current_layer_filters = previous_layer_filters + step
            bn_follows = batch_norm and layer != 1
            if hex:
                layer_list.append(hexagdly.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=1,
                    stride=1,
                    bias=not bn_follows
                ))
            else:
                layer_list.append(nn.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    bias=not bn_follows
                ))

            if layer != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))
                layer_list.append(build_activation(activation))

            previous_layer_filters = current_layer_filters

        if final_activation is not None:
            layer_list.append(build_activation(final_activation))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


##################################################################################################


class ConvProjection_PolicyHead(nn.Module):
    '''Reduces channel count with a single conv layer, flattens the
       remaining spatial features, and projects them to `num_actions`
       logits through a two-layer MLP.'''

    def __init__(
        self,
        width: int,
        num_actions: int,
        dense_layer_neurons: int = 256,
        conv_layer_channels: int = 32,
        activation: str = "relu",
        final_activation: Optional[str] = None,
        batch_norm: bool = False,
        hex: bool = False
    ) -> None:
        super().__init__()

        layer_list: list[nn.Module] = []

        if hex:
            layer_list.append(hexagdly.Conv2d(
                in_channels=width,
                out_channels=conv_layer_channels,
                kernel_size=1,
                stride=1,
                bias=not batch_norm
            ))
        else:
            layer_list.append(nn.Conv2d(
                in_channels=width,
                out_channels=conv_layer_channels,
                kernel_size=3,
                stride=1,
                padding='same',
                bias=not batch_norm
            ))

        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=conv_layer_channels))
        layer_list.append(nn.Flatten())
        layer_list.append(build_activation(activation))
        layer_list.append(nn.LazyLinear(dense_layer_neurons))
        layer_list.append(build_activation(activation))
        layer_list.append(nn.Linear(
            in_features=dense_layer_neurons,
            out_features=num_actions
        ))
        if final_activation is not None:
            layer_list.append(build_activation(final_activation))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


##################################################################################################


class LinearReduce_PolicyHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 3,
        activation: str = "relu",
        final_activation: Optional[str] = None
    ) -> None:
        super().__init__()

        layer_list: list[nn.Module] = []

        delta = out_features - in_features
        step = delta / num_layers
        previous_layer_features = in_features

        for layer in range(num_layers, 0, -1):
            if layer == 1:
                current_layer_features = out_features
            else:
                current_layer_features = previous_layer_features + step

            layer_list.append(nn.Linear(
                max(1, int(previous_layer_features)),
                max(1, int(current_layer_features))
            ))

            if layer != 1:
                layer_list.append(build_activation(activation))

            previous_layer_features = current_layer_features

        if final_activation is not None:
            layer_list.append(build_activation(final_activation))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
