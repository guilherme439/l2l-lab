from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from l2l_lab.rllib.modules.RandomRLModule import RandomRLModule

if TYPE_CHECKING:
    from l2l_lab.configs.training.PolicyConfig import PolicyConfig


class PolicySampler:
    
    def __init__(self, policy_weights: Dict[str, float]):
        self.policies = list(policy_weights.keys())
        self.weights = [policy_weights[p] for p in self.policies]
        self._normalize_weights()
    
    def _normalize_weights(self):
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
    
    def sample(self) -> str:
        return random.choices(self.policies, weights=self.weights, k=1)[0]
    
    def update_weights(self, policy_weights: Dict[str, float]):
        self.policies = list(policy_weights.keys())
        self.weights = [policy_weights[p] for p in self.policies]
        self._normalize_weights()


def create_policy_mapping_fn(sampler: PolicySampler):
    def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
        return sampler.sample()
    return policy_mapping_fn


def build_multi_policy_spec(
    base_spec: RLModuleSpec,
    policy_config: PolicyConfig,
) -> MultiRLModuleSpec:
    specs = {
        "main_policy": base_spec,
    }
    
    if policy_config.random_policy_ratio > 0:
        specs["random_policy"] = RLModuleSpec(
            module_class=RandomRLModule,
            observation_space=base_spec.observation_space,
            action_space=base_spec.action_space,
            model_config={},
            inference_only=True,
        )
    
    for i in range(policy_config.number_previous_policies):
        policy_name = f"checkpoint_{i}"
        specs[policy_name] = RLModuleSpec(
            module_class=base_spec.module_class,
            observation_space=base_spec.observation_space,
            action_space=base_spec.action_space,
            model_config=base_spec.model_config,
            inference_only=True,
        )
    
    return MultiRLModuleSpec(rl_module_specs=specs)


def load_checkpoint_weights_into_policy(algo, policy_name: str, checkpoint_path: Path):
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    backbone_state_dict = checkpoint.get("backbone_state_dict")
    
    if backbone_state_dict is None:
        print(f"Warning: No backbone_state_dict in checkpoint: {checkpoint_path}")
        return False
    
    try:
        rl_module = algo.get_module(policy_name)
    except KeyError:
        print(f"Warning: Policy {policy_name} not found in algorithm")
        return False
    
    if rl_module is None:
        print(f"Warning: Policy {policy_name} returned None")
        return False
    
    if not hasattr(rl_module, 'backbone'):
        print(f"Warning: Policy {policy_name} has no backbone attribute")
        return False
    
    rl_module.backbone.load_state_dict(backbone_state_dict)
    rl_module.eval()
    return True
