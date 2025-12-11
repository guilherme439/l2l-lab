from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import gymnasium as gym
import torch
import torch.nn as nn

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import (
    AddObservationsFromEpisodesToBatch,
)
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner.add_next_observations_from_episodes_to_train_batch import (
    AddNextObservationsFromEpisodesToTrainBatch,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import EpisodeType

if TYPE_CHECKING:
    from ray.rllib.core.learner.torch.torch_learner import TorchLearner
    from ray.rllib.utils.typing import ModuleID

POLICY_MODULE_ID = "shared_policy"
ICM_MODULE_ID = "_intrinsic_curiosity_model"

INTRINSIC_REWARD_COEFF = 0.05
FORWARD_LOSS_WEIGHT = 0.2
FEATURE_DIM = 256
CONV_FILTERS = 32
ICM_LR = 0.0005


class PPOLearnerWithICM(PPOTorchLearner):
    
    def build(self) -> None:
        super().build()
        self._add_icm_connectors()
    
    def _add_icm_connectors(self) -> None:
        learner_config_dict = self.config.learner_config_dict
        
        assert (
            len(self.module) == 2
            and POLICY_MODULE_ID in self.module
            and ICM_MODULE_ID in self.module
        ), f"Expected modules [{POLICY_MODULE_ID}, {ICM_MODULE_ID}], got {list(self.module.keys())}"
        
        if (
            "forward_loss_weight" not in learner_config_dict
            or "intrinsic_reward_coeff" not in learner_config_dict
        ):
            raise KeyError(
                "learner_config_dict must contain 'forward_loss_weight' and 'intrinsic_reward_coeff'"
            )
        
        if self.config.add_default_connectors_to_learner_pipeline:
            self._learner_connector.insert_after(
                AddObservationsFromEpisodesToBatch,
                AddNextObservationsFromEpisodesToTrainBatch(),
            )
            self._learner_connector.insert_after(
                NumpyToTensor,
                ICMConnector(intrinsic_reward_coeff=learner_config_dict["intrinsic_reward_coeff"]),
            )


class ICMConnector(ConnectorV2):
    
    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float,
        **kwargs,
    ):
        super().__init__(input_observation_space, input_action_space)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
    
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Any,
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        assert POLICY_MODULE_ID in batch and ICM_MODULE_ID not in batch
        assert Columns.OBS in batch[POLICY_MODULE_ID] and Columns.NEXT_OBS in batch[POLICY_MODULE_ID]
        
        with torch.no_grad():
            fwd_out = rl_module[ICM_MODULE_ID].forward_train(batch[POLICY_MODULE_ID])
        
        batch[POLICY_MODULE_ID][Columns.REWARDS] += (
            self.intrinsic_reward_coeff * fwd_out[Columns.INTRINSIC_REWARDS]
        )
        batch[ICM_MODULE_ID] = batch[POLICY_MODULE_ID]
        
        return batch


class ConvICM(TorchRLModule, SelfSupervisedLossAPI):
    
    @override(TorchRLModule)
    def setup(self):
        cfg = self.model_config
        
        self.obs_space_format = cfg.get("obs_space_format", "channels_first")
        obs_shape = self.observation_space["observation"].shape
        
        if self.obs_space_format == "channels_first":
            in_channels, h, w = obs_shape[0], obs_shape[1], obs_shape[2]
        else:
            h, w, in_channels = obs_shape[0], obs_shape[1], obs_shape[2]
        
        feature_dim = cfg.get("feature_dim", 256)
        conv_filters = cfg.get("conv_filters", 32)
        
        self._feature_net = nn.Sequential(
            nn.Conv2d(in_channels, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_filters * h * w, feature_dim),
            nn.ReLU(),
        )
        
        self._inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n),
        )
        
        self._forward_net = nn.Sequential(
            nn.Linear(feature_dim + self.action_space.n, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
    
    def _preprocess_obs(self, obs):
        obs = obs.float()
        if self.obs_space_format == "channels_last":
            obs = obs.permute(0, 3, 1, 2)
        return obs
    
    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        obs = self._preprocess_obs(batch[Columns.OBS]["observation"])
        next_obs = self._preprocess_obs(batch[Columns.NEXT_OBS]["observation"])
        
        combined = torch.cat([obs, next_obs], dim=0)
        phis = self._feature_net(combined)
        phi, next_phi = torch.chunk(phis, 2)
        
        actions_onehot = one_hot(batch[Columns.ACTIONS].long(), self.action_space).float()
        predicted_next_phi = self._forward_net(torch.cat([phi, actions_onehot], dim=-1))
        
        forward_l2_norm_sqrt = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )
        
        return {
            Columns.INTRINSIC_REWARDS: forward_l2_norm_sqrt,
            "phi": phi,
            "next_phi": next_phi,
        }
    
    @override(SelfSupervisedLossAPI)
    def compute_self_supervised_loss(
        self,
        *,
        learner: TorchLearner,
        module_id: ModuleID,
        config: AlgorithmConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        module = learner.module[module_id].unwrapped()
        
        forward_loss = torch.mean(fwd_out[Columns.INTRINSIC_REWARDS])
        
        dist_inputs = module._inverse_net(
            torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1)
        )
        action_dist = module.get_train_action_dist_cls().from_logits(dist_inputs)
        inverse_loss = -torch.mean(action_dist.logp(batch[Columns.ACTIONS]))
        
        forward_weight = config.learner_config_dict.get("forward_loss_weight", 0.2)
        total_loss = forward_weight * forward_loss + (1.0 - forward_weight) * inverse_loss
        
        learner.metrics.log_dict(
            {
                "mean_intrinsic_rewards": forward_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
            },
            key=module_id,
            window=1,
        )
        
        return total_loss
    
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError("ConvICM should only be used for training")


def build_icm_rl_module_spec(
    base_spec: MultiRLModuleSpec,
    obs_space,
    act_space,
    obs_space_format: str,
) -> MultiRLModuleSpec:
    return MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": base_spec.rl_module_specs["shared_policy"],
            ICM_MODULE_ID: RLModuleSpec(
                module_class=ConvICM,
                observation_space=obs_space,
                action_space=act_space,
                learner_only=True,
                model_config={
                    "feature_dim": FEATURE_DIM,
                    "conv_filters": CONV_FILTERS,
                    "obs_space_format": obs_space_format,
                },
            ),
        },
    )


def build_icm_training_kwargs(curiosity_coeff: float = INTRINSIC_REWARD_COEFF) -> Dict[str, Any]:
    return {
        "learner_class": PPOLearnerWithICM,
        "learner_config_dict": {
            "intrinsic_reward_coeff": curiosity_coeff,
            "forward_loss_weight": FORWARD_LOSS_WEIGHT,
        },
    }


def build_icm_rl_module_kwargs(base_spec: MultiRLModuleSpec, obs_space, act_space) -> Dict[str, Any]:
    return {
        "rl_module_spec": build_icm_rl_module_spec(
            base_spec,
            obs_space,
            act_space,
            base_spec.rl_module_specs["shared_policy"].model_config.get(
                "obs_space_format", "channels_first"
            ),
        ),
        "algorithm_config_overrides_per_module": {
            ICM_MODULE_ID: AlgorithmConfig.overrides(lr=ICM_LR)
        },
    }
