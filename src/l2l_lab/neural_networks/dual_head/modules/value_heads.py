from __future__ import annotations

from typing import Optional

import hexagdly
from torch import nn

from l2l_lab.neural_networks.utils.builders import build_activation


class ConvReduce_ValueHead(nn.Module):
    '''Several conv layers that progressively reduce the amount of filters,
       followed by a global average pooling layer'''

    def __init__(
        self,
        width: int,
        num_reduce_layers: int = 4,
        activation: str = "tanh",
        final_activation: Optional[str] = None,
        batch_norm: bool = False,
        hex: bool = False
    ) -> None:
        super().__init__()

        value_head_layers: list[nn.Module] = []
        final_layer_filters = 1

        delta = final_layer_filters - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers, 0, -1):
            current_layer_filters = previous_layer_filters + step
            bn_follows = batch_norm and layer != 1
            if hex:
                conv = hexagdly.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=1,
                    stride=1,
                    bias=not bn_follows
                )
            else:
                conv = nn.Conv2d(
                    in_channels=int(previous_layer_filters),
                    out_channels=int(current_layer_filters),
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    bias=not bn_follows
                )
            value_head_layers.append(conv)

            if layer != 1:
                if batch_norm:
                    value_head_layers.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))
                value_head_layers.append(build_activation(activation))

            previous_layer_filters = current_layer_filters

        value_head_layers.append(nn.AdaptiveAvgPool3d(1))
        value_head_layers.append(nn.Flatten())
        if final_activation is not None:
            value_head_layers.append(build_activation(final_activation))

        self.layers = nn.Sequential(*value_head_layers)

    def forward(self, x):
        return self.layers(x)


# ------------------------------ #


class ConvProjection_ValueHead(nn.Module):
    '''Reduces channel count with a single conv layer, flattens the
       remaining spatial features, and projects them to a scalar value
       through a two-layer MLP.'''

    def __init__(
        self,
        width: int,
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
            out_features=1
        ))
        if final_activation is not None:
            layer_list.append(build_activation(final_activation))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


##################################################################################################


class LinearReduce_ValueHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_layers: int = 3,
        activation: str = "tanh",
        final_activation: Optional[str] = None
    ) -> None:
        super().__init__()

        layer_list: list[nn.Module] = []

        delta = 1 - in_features
        step = delta / num_layers
        previous_layer_features = in_features

        for layer in range(num_layers, 0, -1):
            if layer == 1:
                current_layer_features = 1
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
