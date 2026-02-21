from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig

from .base import BaseAlgorithmTrainer

if TYPE_CHECKING:
    from configs.definition.training.TrainingConfig import TrainingConfig


class IMPALATrainer(BaseAlgorithmTrainer):

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
    
    @property
    def algorithm_name(self) -> str:
        return "impala"

    def load_from_checkpoint(self, checkpoint_path: Path):
        return IMPALA.from_checkpoint(str(checkpoint_path.absolute()))
    
    def extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        env_runners = result.get("env_runners", {})
        learner_stats = result.get("learners", {}).get("shared_policy", {})
        
        return {
            "episode_len_mean": env_runners.get("episode_len_mean", 0) or 0,
            "total_loss": learner_stats.get("total_loss", 0) or 0,
            "policy_loss": learner_stats.get("mean_pi_loss", 0) or 0,
            "vf_loss": learner_stats.get("mean_vf_loss", 0) or 0,
            "entropy": learner_stats.get("entropy"),
            "learning_rate": learner_stats.get("default_optimizer_learning_rate"),
        }
    
    def build_config(self, env_name: str, obs_space_format, obs_space, act_space) -> IMPALAConfig:
        cfg = self.config.algorithm.config
        
        return (
            IMPALAConfig()
            .environment(
                env=env_name,
                disable_env_checking=True,
            )
            .env_runners(
                num_env_runners=cfg.num_env_runners,
                rollout_fragment_length=cfg.rollout_fragment_length,
            )
            .training(
                lr=cfg.lr,
                gamma=cfg.gamma,
                entropy_coeff=cfg.entropy_coeff,
                vf_loss_coeff=cfg.vf_loss_coeff,
                grad_clip=cfg.grad_clip,
                vtrace=cfg.vtrace,
                vtrace_clip_rho_threshold=cfg.vtrace_clip_rho_threshold,
                vtrace_clip_pg_rho_threshold=cfg.vtrace_clip_pg_rho_threshold,
            )
            .learners(
                num_learners=cfg.num_learners,
            )
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .rl_module(
                rl_module_spec=self.get_rl_module_spec(obs_space, obs_space_format, act_space),
            )
            .framework("torch")
            .resources(num_gpus=0)
            .debugging(log_level="WARN")
        )
    
    
