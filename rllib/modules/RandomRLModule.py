from __future__ import annotations

from typing import Any, Dict, Type

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import FLOAT_MIN


class RandomRLModule(TorchRLModule):
    
    @override(TorchRLModule)
    def setup(self):
        self._dummy = torch.nn.Parameter(torch.zeros(1))
    
    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self) -> Type[Distribution]:
        return TorchCategorical
    
    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> Type[Distribution]:
        return TorchCategorical
    
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._uniform_logits(batch)
    
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._uniform_logits(batch)
    
    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._uniform_logits(batch)
    
    def _uniform_logits(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obs_dict = batch[Columns.OBS]
        action_mask = obs_dict["action_mask"]
        
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask)
        
        logits = torch.zeros_like(action_mask, dtype=torch.float32)
        invalid = (action_mask == 0)
        logits = logits.masked_fill(invalid, FLOAT_MIN)
        
        return {Columns.ACTION_DIST_INPUTS: logits}
