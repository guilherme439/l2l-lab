from typing import Any, Dict, Type, TypedDict

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import nn


class DualHeadModelConfig(TypedDict):
    network_class: Type[nn.Module]
    network_kwargs: Dict[str, Any]


class DualHeadRLModule(TorchRLModule, ValueFunctionAPI):
    model_config: DualHeadModelConfig
    
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "DualHeadRLModule requires a Dict observation space with "
                "'observation' and 'action_mask' keys."
            )
        
        inner_obs_space = self.observation_space["observation"]
        
        network_class = self.model_config["network_class"]
        network_kwargs = self.model_config.get("network_kwargs", {})
        
        obs_shape = inner_obs_space.shape
        in_channels = obs_shape[0]
        rows, cols = obs_shape[1], obs_shape[2]
        
        policy_channels = self.action_space.n // (rows * cols)
        
        self.backbone = network_class(
            in_channels=in_channels,
            policy_channels=policy_channels,
            **network_kwargs,
        )
    
    def _forward(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        obs_dict = batch[Columns.OBS]
        obs = obs_dict["observation"].float()
        action_mask = obs_dict["action_mask"]
        
        policy_logits, value = self.backbone(obs)
        
        policy_logits = policy_logits.reshape(policy_logits.shape[0], -1)
        
        inf_mask = torch.clamp(torch.log(action_mask.float() + 1e-10), min=FLOAT_MIN)
        policy_logits = policy_logits + inf_mask
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
            Columns.VF_PREDS: value,
        }
    
    @override(TorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, TensorType], embeddings: Any = None
    ) -> TensorType:
        obs_dict = batch[Columns.OBS]
        if isinstance(obs_dict, dict):
            obs = obs_dict["observation"].float()
        else:
            obs = obs_dict.float()
        
        _, value = self.backbone(obs)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return value
