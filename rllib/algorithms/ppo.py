from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from ray.rllib.algorithms.ppo import PPO, PPOConfig

from .base import BaseAlgorithmTrainer

if TYPE_CHECKING:
    from Trainer import Trainer


class PPOTrainer(BaseAlgorithmTrainer):
    
    def __init__(self, trainer: Trainer):
        super().__init__(trainer)
    
    @property
    def algorithm_name(self) -> str:
        return "ppo"

    def load_from_checkpoint(self, checkpoint_path: Path):
        return PPO.from_checkpoint(str(checkpoint_path.absolute()))
    
    def extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        env_runners = result.get("env_runners", {})
        learner_stats = result.get("learners", {}).get("shared_policy", {})
        
        return {
            "episode_len_mean": env_runners.get("episode_len_mean", 0) or 0,
            "total_loss": learner_stats.get("total_loss", 0) or 0,
            "policy_loss": learner_stats.get("policy_loss", 0) or 0,
            "vf_loss": learner_stats.get("vf_loss", 0) or 0,
            "vf_loss_unclipped": learner_stats.get("vf_loss_unclipped"),
            "entropy": learner_stats.get("entropy"),
            "kl_divergence": learner_stats.get("mean_kl_loss"),
            "vf_explained_var": learner_stats.get("vf_explained_var"),
        }
    
    def build_config(self, env_name: str, obs_space_format, obs_space, act_space, ) -> PPOConfig:
        cfg = self.config.algorithm.config

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
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .framework("torch")
            .resources(num_gpus=0)
            .debugging(log_level="WARN")
        )

        if cfg.use_curiosity:
            curiosity_coeff = cfg.curiosity_coeff
            self._with_curiosity(config, obs_space, obs_space_format, act_space, curiosity_coeff)
        else:
            self._without_curiosity(config, obs_space, obs_space_format, act_space)
        
        return config
    
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
    
    
