
import hexagdly

from torch import nn


class Reduce_ValueHead(nn.Module):
    '''Several conv layers that progressively reduce the amount of filters,
       followed by a global average pooling layer'''

    def __init__(self, width, num_reduce_layers=4, activation="tanh", batch_norm=False, hex=True):
        super().__init__()

        value_head_layers = []
        final_layer_filters = 1

        delta = final_layer_filters - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers,0,-1):
            current_layer_filters = previous_layer_filters + step
            if hex:
                conv = hexagdly.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=1, stride=1, bias=False)
            else:
                conv = nn.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=3, stride=1, padding='same', bias=False)
            value_head_layers.append(conv)

            if layer != 1:
                if batch_norm:
                    value_head_layers.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))

                if activation == "tanh":
                    value_head_layers.append(nn.Tanh())
                elif activation == "relu":
                    value_head_layers.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

            previous_layer_filters = current_layer_filters

        value_head_layers.append(nn.AdaptiveAvgPool3d(1))
        value_head_layers.append(nn.Flatten())
        value_head_layers.append(nn.Tanh())

        self.layers = nn.Sequential(*value_head_layers)



    def forward(self, x):
        out = self.layers(x)
        return out
    

# ------------------------------ #
 
class Dense_ValueHead(nn.Module):

    def __init__(self, width, dense_layer_neurons=256, conv_layer_channels=32, batch_norm=False, hex=True):
        super().__init__()        
        
        layer_list = []
        
        if hex:
            layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=conv_layer_channels, kernel_size=1, stride=1, bias=False))
        else:
            layer_list.append(nn.Conv2d(in_channels=width, out_channels=conv_layer_channels, kernel_size=3, stride=1, padding='same', bias=False))

        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=conv_layer_channels))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.ReLU())
        layer_list.append(nn.LazyLinear(dense_layer_neurons, bias=False))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(in_features=dense_layer_neurons, out_features=1, bias=False))
        layer_list.append(nn.Tanh())
        

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    

##################################################################################################


class ReduceMLP_ValueHead(nn.Module):

    def __init__(self, in_features, num_layers=3):
        super().__init__()

        layer_list = []

        delta = 1 - in_features
        step = delta / num_layers
        previous_layer_features = in_features

        for layer in range(num_layers, 0, -1):
            current_layer_features = previous_layer_features + step

            layer_list.append(nn.Linear(
                max(1, int(previous_layer_features)),
                max(1, int(current_layer_features)),
            ))

            if layer != 1:
                layer_list.append(nn.Tanh())

            previous_layer_features = current_layer_features

        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
