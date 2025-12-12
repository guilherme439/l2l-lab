from typing import Any, Dict, Type, TypedDict

import torch
import gymnasium as gym
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import nn


class MLPDualHeadModelConfig(TypedDict):
    network_class: Type[nn.Module]
    network_kwargs: Dict[str, Any]


class MLPDualHeadRLModule(TorchRLModule, ValueFunctionAPI):
    
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        
        network_class = self.model_config["network_class"]
        network_kwargs = self.model_config.get("network_kwargs", {})
        
        out_features = self.action_space.n
        
        self.backbone = network_class(
            out_features=out_features,
            **network_kwargs,
        )

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "ConvDualHeadRLModule requires a Dict observation space with "
                "'observation' and 'action_mask' keys."
            )

        inner_obs_space = self.observation_space["observation"]
        dummy_obs_shape = (1,) + inner_obs_space.shape
        dummy_obs = torch.zeros(dummy_obs_shape, dtype=torch.float32)

        with torch.no_grad():
            _ = self.backbone(dummy_obs)
    
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        obs_dict: Dict[str, Any] = batch[Columns.OBS]
        obs = obs_dict["observation"].float()

        action_mask = obs_dict["action_mask"]
        invalid = (action_mask == 0)
        policy_logits, value = self.backbone(obs)
        policy_logits = policy_logits.masked_fill(invalid, FLOAT_MIN)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return {Columns.ACTION_DIST_INPUTS: policy_logits, Columns.VF_PREDS: value}

    def _forward_policy_only(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        obs_dict: Dict[str, Any] = batch[Columns.OBS]
        obs = obs_dict["observation"].float()
        action_mask = obs_dict["action_mask"]
        invalid = (action_mask == 0)

        embeddings = self.backbone.forward_trunk(obs)
        policy_logits = self.backbone.policy_head(embeddings)
        policy_logits = policy_logits.masked_fill(invalid, FLOAT_MIN)
        
        return {
            Columns.ACTION_DIST_INPUTS: policy_logits,
            Columns.EMBEDDINGS: embeddings
        }
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        return self._forward_policy_only(batch, **kwargs)
    
    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Any = None) -> TensorType:
        if embeddings is not None:
            value = self.backbone.value_head(embeddings)
        else:
            obs_dict: Dict[str, Any] = batch[Columns.OBS]
            obs = obs_dict["observation"].float()
            _, value = self.backbone(obs)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return value
