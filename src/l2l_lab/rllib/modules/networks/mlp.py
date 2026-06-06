import math
from typing import Any, TypedDict

import gymnasium as gym
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

from l2l_lab.configs.training.network import network_config_from_dict
from l2l_lab.neural_networks.utils.builders import build_network


class MLPDualHeadModelConfig(TypedDict):
    network_config: dict[str, Any]


class MLPDualHeadRLModule(TorchRLModule, ValueFunctionAPI):

    @override(TorchRLModule)
    def setup(self):
        super().setup()

        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError(
                "MLPDualHeadRLModule requires a Dict observation space with "
                "'observation' and 'action_mask' keys."
            )

        network_cfg = network_config_from_dict(self.model_config["network_config"])

        num_actions = self.action_space.n
        inner_obs_space = self.observation_space["observation"]
        input_features = int(math.prod(inner_obs_space.shape))

        self.backbone = build_network(
            network_cfg,
            input_features=input_features,
            num_actions=num_actions,
        )
    
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, TensorType]:
        obs_dict: dict[str, Any] = batch[Columns.OBS]
        obs = obs_dict["observation"].float()

        action_mask = obs_dict["action_mask"]
        invalid = (action_mask == 0)
        policy_logits, value = self.backbone(obs)
        policy_logits = policy_logits.masked_fill(invalid, FLOAT_MIN)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return {Columns.ACTION_DIST_INPUTS: policy_logits, Columns.VF_PREDS: value}

    def _forward_policy_only(self, batch: dict[str, Any], **kwargs) -> dict[str, TensorType]:
        obs_dict: dict[str, Any] = batch[Columns.OBS]
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
    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, TensorType]:
        return self._forward(batch, **kwargs)
    
    @override(TorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, TensorType]:
        return self._forward_policy_only(batch, **kwargs)
    
    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> TensorType:
        if embeddings is not None:
            value = self.backbone.value_head(embeddings)
        else:
            obs_dict: dict[str, Any] = batch[Columns.OBS]
            obs = obs_dict["observation"].float()
            _, value = self.backbone(obs)
        
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return value
