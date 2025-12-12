from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from .base import BaseAlgorithmTrainer
from .multi_policy import (
    PolicySampler,
    build_multi_policy_spec,
    create_policy_mapping_fn,
    load_checkpoint_weights_into_policy,
)

if TYPE_CHECKING:
    from Trainer import Trainer


class PPOTrainer(BaseAlgorithmTrainer):
    
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
        self.policy_sampler: Optional[PolicySampler] = None
        self.available_checkpoints: List[int] = []
    
    @property
    def algorithm_name(self) -> str:
        return "ppo"

    def load_from_checkpoint(self, checkpoint_path: Path):
        return PPO.from_checkpoint(str(checkpoint_path.absolute()))
    
    def extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        env_runners = result.get("env_runners", {})
        policy_cfg = self.config.algorithm.config.policy
        policy_name = "main_policy" if policy_cfg and policy_cfg.use_multiple_policies else "shared_policy"
        learner_stats = result.get("learners", {}).get(policy_name, {})
        icm_stats = result.get("learners", {}).get("_intrinsic_curiosity_model", {})
        
        metrics = {
            "episode_len_mean": env_runners.get("episode_len_mean", 0) or 0,
            "episode_reward_mean": env_runners.get("episode_reward_mean"),
            "total_loss": learner_stats.get("total_loss", 0) or 0,
            "policy_loss": learner_stats.get("policy_loss", 0) or 0,
            "vf_loss": learner_stats.get("vf_loss", 0) or 0,
            "vf_loss_unclipped": learner_stats.get("vf_loss_unclipped"),
            "entropy": learner_stats.get("entropy"),
            "kl_divergence": learner_stats.get("mean_kl_loss"),
            "vf_explained_var": learner_stats.get("vf_explained_var"),
            "learning_rate": learner_stats.get("default_optimizer_learning_rate"),
        }
        
        if icm_stats:
            metrics["intrinsic_reward_mean"] = icm_stats.get("mean_intrinsic_rewards")
            metrics["icm_forward_loss"] = icm_stats.get("forward_loss")
            metrics["icm_inverse_loss"] = icm_stats.get("inverse_loss")
        
        return metrics
    
    def build_config(self, env_name: str, obs_space_format, obs_space, act_space) -> PPOConfig:
        cfg = self.config.algorithm.config
        policy_cfg = self.config.algorithm.config.policy

        config = (
            PPOConfig()
            .environment(
                env=env_name,
                disable_env_checking=True,
            )
            .env_runners(
                num_env_runners=cfg.num_env_runners,
                rollout_fragment_length=cfg.rollout_fragment_length,
            )
            .training(
                train_batch_size_per_learner=cfg.train_batch_size_per_learner,
                minibatch_size=cfg.minibatch_size,
                num_epochs=cfg.num_epochs,
                lr=cfg.lr,
                gamma=cfg.gamma,
                entropy_coeff=cfg.entropy_coeff,
                vf_loss_coeff=cfg.vf_loss_coeff,
                clip_param=cfg.clip_param,
                use_kl_loss=cfg.use_kl_loss,
                kl_coeff=cfg.kl_coeff,
                kl_target=cfg.kl_target,
            )
            .framework("torch")
            .resources(num_gpus=0)
            .debugging(log_level="WARN")
        )

        if policy_cfg and policy_cfg.use_multiple_policies:
            self._setup_multi_policy(config, obs_space, obs_space_format, act_space, cfg.use_curiosity, cfg.curiosity_coeff if cfg.use_curiosity else 0.0)
        else:
            config.multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            if cfg.use_curiosity:
                self._with_curiosity(config, obs_space, obs_space_format, act_space, cfg.curiosity_coeff)
            else:
                self._without_curiosity(config, obs_space, obs_space_format, act_space)
        
        return config
    
    def _setup_multi_policy(
        self,
        config: PPOConfig,
        obs_space,
        obs_space_format,
        act_space,
        use_curiosity: bool,
        curiosity_coeff: float,
    ):
        policy_cfg = self.config.algorithm.config.policy
        
        policy_weights = policy_cfg.get_policy_weights(len(self.available_checkpoints))
        self.policy_sampler = PolicySampler(policy_weights)
        
        base_rl_module_spec = self._get_base_rl_module_spec(obs_space, obs_space_format, act_space)
        multi_spec = build_multi_policy_spec(base_rl_module_spec, policy_cfg)
        
        policy_names = set(multi_spec.rl_module_specs.keys())
        
        config.multi_agent(
            policies=policy_names,
            policy_mapping_fn=create_policy_mapping_fn(self.policy_sampler),
            policies_to_train={"main_policy"},
        )
        
        if use_curiosity:
            raise NotImplementedError("Curiosity (ICM) is not yet supported with multi-policy setup")
        
        config.rl_module(rl_module_spec=multi_spec)
    
    def _get_base_rl_module_spec(self, obs_space, obs_space_format, act_space) -> RLModuleSpec:
        from rllib.modules.networks.conv import ConvDualHeadRLModule
        
        network_class = self.config.network.get_network_class()
        adapter_class = self.config.network.get_adapter_class()
        
        model_config = {
            "network_class": network_class,
            "network_kwargs": self.config.network.to_kwargs(),
            "architecture": self.config.network.architecture,
        }
        
        if adapter_class == ConvDualHeadRLModule:
            model_config["obs_space_format"] = obs_space_format
        
        return RLModuleSpec(
            module_class=adapter_class,
            observation_space=obs_space,
            action_space=act_space,
            model_config=model_config,
        )
    
    def update_opponent_policies(self, model_dir: Path, new_checkpoint: int):
        policy_cfg = self.config.algorithm.config.policy
        if not policy_cfg or not policy_cfg.use_multiple_policies:
            return
        
        if new_checkpoint not in self.available_checkpoints:
            self.available_checkpoints.append(new_checkpoint)
        
        n_prev = policy_cfg.number_previous_policies
        
        if n_prev == 0:
            return
        
        recent = sorted(self.available_checkpoints, reverse=True)[:n_prev]
        
        for slot_idx, cp_iter in enumerate(recent):
            policy_name = f"checkpoint_{slot_idx}"
            cp_path = model_dir / "checkpoints" / str(cp_iter) / "model.cp"
            
            try:
                if self.algo is not None:
                    load_checkpoint_weights_into_policy(self.algo, policy_name, cp_path)
            except Exception as e:
                print(f"Warning: Could not load checkpoint {cp_iter} into {policy_name}: {e}")
        
        policy_weights = policy_cfg.get_policy_weights(len(self.available_checkpoints))
        if self.policy_sampler:
            self.policy_sampler.update_weights(policy_weights)
    
    def _without_curiosity(self, rllib_config: PPOConfig, obs_space, obs_space_format, act_space) -> PPOConfig:
        return rllib_config.rl_module(rl_module_spec=self.get_rl_module_spec(obs_space, obs_space_format, act_space))
    
    def _with_curiosity(self, rllib_config: PPOConfig, obs_space, obs_space_format, act_space, curiosity_coeff: float) -> PPOConfig:
        from .icm import build_icm_training_kwargs, build_icm_rl_module_kwargs
        
        base_spec = self.get_rl_module_spec(obs_space, obs_space_format, act_space)
        return (
            rllib_config
            .training(**build_icm_training_kwargs(curiosity_coeff=curiosity_coeff))
            .rl_module(**build_icm_rl_module_kwargs(base_spec, obs_space, act_space))
        )
    
    
