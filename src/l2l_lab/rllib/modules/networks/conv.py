from typing import Any, Dict, Literal, Type, TypedDict

import gymnasium as gym
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import nn

from l2l_lab.configs.training.NetworkConfig import NetworkConfig


class ConvDualHeadModelConfig(TypedDict):
    network_class: Type[nn.Module]
    network_kwargs: Dict[str, Any]
    obs_space_format: Literal["channels_first", "channels_last"]


class ConvDualHeadRLModule(TorchRLModule, ValueFunctionAPI):
    
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "ConvDualHeadRLModule requires a Dict observation space with "
                "'observation' and 'action_mask' keys."
            )
        
        self.obs_space_format = self.model_config.get("obs_space_format")
        inner_obs_space = self.observation_space["observation"]
        
        network_class = self.model_config["network_class"]
        network_kwargs = self.model_config.get("network_kwargs", {})

        # FIXME: this code is a mess and needs to be improved    
        obs_shape = inner_obs_space.shape
        if self.obs_space_format == "channels_first":
            in_channels, h, w = obs_shape[0], obs_shape[1], obs_shape[2]
        elif self.obs_space_format == "channels_last":
            h, w, in_channels = obs_shape[0], obs_shape[1], obs_shape[2]
        else:
            raise ValueError(f"Unsupported obs_space_format: {self.obs_space_format}")

        num_actions = self.action_space.n
        NetworkConfig(
            architecture=network_class.__name__,
            kwargs=network_kwargs,
        ).validate_for_env((in_channels, h, w), num_actions)

        self.backbone = network_class(in_channels=in_channels, num_actions=num_actions, **network_kwargs)
    
    def _preprocess_obs(self, obs: TensorType) -> TensorType:
        if self.obs_space_format == "channels_last":
            return obs.permute(0, 3, 1, 2)
        return obs
    
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        obs_dict: Dict[str, Any] = batch[Columns.OBS]
        obs = self._preprocess_obs(obs_dict["observation"].float())

        action_mask = obs_dict["action_mask"]
        invalid = (action_mask == 0)
        policy_logits, value = self.backbone(obs)
        policy_logits = policy_logits.reshape(policy_logits.shape[0], -1)
        policy_logits = policy_logits.masked_fill(invalid, FLOAT_MIN)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return {Columns.ACTION_DIST_INPUTS: policy_logits, Columns.VF_PREDS: value}

    def _forward_policy_only(self, batch: Dict[str, Any], **kwargs) -> Dict[str, TensorType]:
        obs_dict: Dict[str, Any] = batch[Columns.OBS]
        obs = self._preprocess_obs(obs_dict["observation"].float())
        action_mask = obs_dict["action_mask"]
        invalid = (action_mask == 0)

        embeddings = self.backbone.forward_trunk(obs)
        policy_logits = self.backbone.policy_head(embeddings)
        policy_logits = policy_logits.reshape(policy_logits.shape[0], -1)
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
            obs = self._preprocess_obs(obs_dict["observation"].float())
            _, value = self.backbone(obs)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return value
