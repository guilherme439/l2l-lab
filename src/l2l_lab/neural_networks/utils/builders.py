from __future__ import annotations

from typing import Optional

from torch import nn

from l2l_lab.configs.training.network import (BaseNetworkConfig,
                                               BasePolicyHeadConfig,
                                               BaseValueHeadConfig,
                                               ConvNetConfig,
                                               ConvProjectionPolicyHeadConfig,
                                               ConvProjectionValueHeadConfig,
                                               ConvReducePolicyHeadConfig,
                                               ConvReduceValueHeadConfig,
                                               LinearReducePolicyHeadConfig,
                                               LinearReduceValueHeadConfig,
                                               MLPNetConfig, RecurrentNetConfig,
                                               ResNetConfig, SNNetConfig)




def build_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name!r}")


def build_policy_head(
    cfg: BasePolicyHeadConfig,
    *,
    num_filters: Optional[int] = None,
    num_actions: Optional[int] = None,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None
) -> nn.Module:
    from l2l_lab.neural_networks.dual_head.modules.policy_heads import (
        ConvProjection_PolicyHead, ConvReduce_PolicyHead, LinearReduce_PolicyHead)

    match cfg:
        case ConvProjectionPolicyHeadConfig():
            return ConvProjection_PolicyHead(
                width=num_filters,
                num_actions=num_actions,
                dense_layer_neurons=cfg.dense_layer_neurons,
                conv_layer_channels=cfg.conv_layer_channels,
                activation=cfg.activation,
                final_activation=cfg.final_activation,
                batch_norm=cfg.batch_norm,
                hex=cfg.hex
            )
        case ConvReducePolicyHeadConfig():
            return ConvReduce_PolicyHead(
                width=num_filters,
                policy_channels=cfg.policy_channels,
                num_reduce_layers=cfg.num_reduce_layers,
                activation=cfg.activation,
                final_activation=cfg.final_activation,
                batch_norm=cfg.batch_norm,
                hex=cfg.hex
            )
        case LinearReducePolicyHeadConfig():
            return LinearReduce_PolicyHead(
                in_features=in_features,
                out_features=out_features,
                num_layers=cfg.num_layers,
                activation=cfg.activation,
                final_activation=cfg.final_activation
            )
        case _:
            raise TypeError(f"Unknown policy head config type: {type(cfg).__name__}")


def build_value_head(
    cfg: BaseValueHeadConfig,
    *,
    num_filters: Optional[int] = None,
    in_features: Optional[int] = None
) -> nn.Module:
    from l2l_lab.neural_networks.dual_head.modules.value_heads import (
        ConvProjection_ValueHead, ConvReduce_ValueHead, LinearReduce_ValueHead)

    match cfg:
        case ConvProjectionValueHeadConfig():
            return ConvProjection_ValueHead(
                width=num_filters,
                dense_layer_neurons=cfg.dense_layer_neurons,
                conv_layer_channels=cfg.conv_layer_channels,
                activation=cfg.activation,
                final_activation=cfg.final_activation,
                batch_norm=cfg.batch_norm,
                hex=cfg.hex
            )
        case ConvReduceValueHeadConfig():
            return ConvReduce_ValueHead(
                width=num_filters,
                num_reduce_layers=cfg.num_reduce_layers,
                activation=cfg.activation,
                final_activation=cfg.final_activation,
                batch_norm=cfg.batch_norm,
                hex=cfg.hex
            )
        case LinearReduceValueHeadConfig():
            return LinearReduce_ValueHead(
                in_features=in_features,
                num_layers=cfg.num_layers,
                activation=cfg.activation,
                final_activation=cfg.final_activation
            )
        case _:
            raise TypeError(f"Unknown value head config type: {type(cfg).__name__}")


def build_network(
    cfg: BaseNetworkConfig,
    *,
    in_channels: Optional[int] = None,
    num_actions: Optional[int] = None,
    input_features: Optional[int] = None,
) -> nn.Module:
    from l2l_lab.neural_networks.dual_head.ConvNet import ConvNet
    from l2l_lab.neural_networks.dual_head.MLPNet import MLPNet
    from l2l_lab.neural_networks.dual_head.RecurrentNet import RecurrentNet
    from l2l_lab.neural_networks.dual_head.ResNet import ResNet
    from l2l_lab.neural_networks.dual_head.SNNet import SNNet

    match cfg:
        case ResNetConfig():
            return ResNet(cfg, in_channels=in_channels, num_actions=num_actions)
        case ConvNetConfig():
            return ConvNet(cfg, in_channels=in_channels, num_actions=num_actions)
        case RecurrentNetConfig():
            return RecurrentNet(cfg, in_channels=in_channels, num_actions=num_actions)
        case MLPNetConfig():
            return MLPNet(cfg, input_features=input_features, num_actions=num_actions)
        case SNNetConfig():
            return SNNet(cfg, input_features=input_features, num_actions=num_actions)
        case _:
            raise TypeError(f"Unknown network config type: {type(cfg).__name__}")
