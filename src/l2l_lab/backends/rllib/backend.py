from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, override

import torch
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from torch import nn
from ray.tune.registry import register_env

from l2l_lab._utils.checkpoint import CheckpointUtils
from l2l_lab._utils.common import CommonUtils
from l2l_lab.backends.backend_base import AlgorithmBackend, StepResult
from l2l_lab.envs.registry import create_env
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from l2l_lab.configs.training.training_config import TrainingConfig
    from l2l_lab.rllib.algorithms.base import BaseAlgorithmTrainer


class RLlibBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self.algo: Any = None
        self.algo_trainer: Optional[BaseAlgorithmTrainer] = None
        self._config: Optional[TrainingConfig] = None
        self._input_shape: Optional[tuple] = None
        self._num_actions: Optional[int] = None

    @property
    @override
    def name(self) -> str:
        if self._config:
            return f"rllib_{self._config.backend.algorithm.name}"
        return "rllib"

    @override
    def init(self) -> None:
        import ray
        if not ray.is_initialized():
            logger.info("")
            ray.init(
                ignore_reinit_error=True,
                _system_config={
                    "gcs_rpc_server_reconnect_timeout_s": 120,
                    "gcs_server_request_timeout_seconds": 120,
                    "local_fs_capacity_threshold": 0.99
                },
                object_store_memory=2 * 1024 * 1024 * 1024,
            )
            logger.info("")
    
    @override
    def shutdown(self) -> None:
        if self.algo is not None:
            self.algo.stop()
        import ray
        if ray.is_initialized():
            ray.shutdown()

    @override
    def prepare(self, config: TrainingConfig) -> None:
        self._config = config
        self._register_env(config.env)
        obs_space, act_space = self._get_spaces(config.env)
        self._input_shape = obs_space["observation"].shape
        self._num_actions = act_space.n

        logger.info(f"\nEnvironment Info:")
        logger.info("  Observation space:")
        for key, space in obs_space.spaces.items():
            if hasattr(space, 'shape'):
                logger.info(f"    {key}: shape={space.shape}, dtype={space.dtype}")
            else:
                logger.info(f"    {key}: {space}")
        logger.info(f"  Action space: {act_space}")
        logger.info("")
        logger.info("=" * 70)

        self.algo_trainer = self._get_algorithm_trainer()
        rllib_config = self.algo_trainer.build_config(
            config.env.name, config.env.obs_space_format, obs_space, act_space
        )

        logger.info(f"\nBuilding {config.backend.algorithm.name.upper()} algorithm...\n")
        self.algo = rllib_config.build_algo()
        self.algo_trainer.algo = self.algo

        self._total_iterations = config.backend.algorithm.total_iterations

        logger.info("\n✓ Algorithm built successfully!")

    @override
    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        algo_path = checkpoint_dir / "algo"
        if not algo_path.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_dir}' is missing 'algo'.")
        self.algo.restore_from_path(str(algo_path.absolute()))
        logger.info(f"\n✓ Weights restored from {algo_path}")

    @override
    def init_fresh(self) -> None:
        pass

    @override
    def _get_live_model(self) -> nn.Module:
        return self._get_backbone()

    @override
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> nn.Module:
        backbone = torch.load(CheckpointUtils.get_network_template_path(checkpoint_dir), weights_only=False)
        state_dict = CheckpointUtils.load_checkpoint_file(checkpoint_dir / "weights.pt")
        CheckpointUtils.load_model_state_dict(backbone, state_dict)
        backbone.eval()
        return backbone

    @override
    def get_weight_parameters(self) -> Optional[Iterator]:
        rl_module = self.algo.get_module(self._get_policy_name())
        return rl_module.parameters()

    @override
    def save_final_checkpoint(self, iteration: int) -> Optional[Path]:
        if self._checkpoint_interval <= 0 or self._checkpoint_base_dir is None:
            return None
        checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(iteration)
        self._write_checkpoint(checkpoint_path)
        return checkpoint_path

    @override
    def on_checkpoint_saved(self, model_dir: Path, iteration: int) -> None:
        self.algo_trainer.update_opponent_policies(model_dir, iteration)

    @override
    def train(self) -> None:
        info_interval = self._config.common.info_interval
        try:
            for current_iteration in range(self._starting_iteration, self._total_iterations):
                if self._stop_event.is_set():
                    break
                iterations_completed = current_iteration + 1
                result = self.algo.train()
                metrics = self.algo_trainer.extract_metrics(result)

                self.print_step_info(current_iteration, metrics)
                if CommonUtils.check_interval(iterations_completed, info_interval):
                    self.print_training_info(current_iteration, metrics)

                checkpoint_path: Optional[Path] = None
                if CommonUtils.check_interval(iterations_completed, self._checkpoint_interval):
                    logger.info(f"\nSaving {self.name} checkpoint for iteration {current_iteration}")
                    checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(current_iteration)
                    # Written synchronously inside the step callback,
                    # to garantee that the data saved is for this exact point in time.
                    self._write_checkpoint(checkpoint_path)

                eval_model = self._get_eval_model() if self._needs_snapshot(iterations_completed) else None

                self.step_queue.put(StepResult(
                    iteration=current_iteration,
                    metrics=metrics,
                    checkpoint_path=checkpoint_path,
                    eval_model=eval_model,
                ))
        finally:
            self.step_queue.put(None)

    def _write_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        CheckpointUtils.atomic_write(path / "weights.pt", lambda temp_path: torch.save(self._get_backbone().state_dict(), temp_path))
        self.algo.save_to_path(str((path / "algo").absolute()))

    def _get_backbone(self) -> nn.Module:
        return self.algo.get_module(self._get_policy_name()).backbone

    @override
    def print_step_info(self, iteration: int, metrics: dict[str, Any]) -> None:
        ep_len = metrics.get("episode_len_mean", 0) or 0
        total = self._config.backend.algorithm.total_iterations
        logger.info(f"\n{iteration}/{total} | EpLen: {ep_len:6.1f}\n")

    @override
    def print_training_info(self, iteration: int, metrics: dict[str, Any]) -> None:
        timesteps = metrics.get("timesteps_lifetime", 0)
        curr_lr = metrics.get("learning_rate")

        logger.info("")
        logger.info("  ┌─ Training Info ─────────────────────────────")
        logger.info(f"  │ Timesteps: {timesteps:,}")
        if curr_lr is not None:
            logger.info(f"  │ Learning Rate: {curr_lr:.2e}")
        logger.info("  └──────────────────────────────────────────────")
        logger.info("")

    def _register_env(self, env_config) -> None:
        def env_creator(config: dict):
            env = create_env(env_config.name, **env_config.kwargs)
            return PettingZooEnv(env)
        register_env(env_config.name, env_creator)

    def _get_spaces(self, env_config):
        env = create_env(env_config.name, **env_config.kwargs)
        wrapped = PettingZooEnv(env)
        first_agent = list(wrapped.observation_space.keys())[0]
        obs_space = wrapped.observation_space[first_agent]
        act_space = wrapped.action_space[first_agent]
        return obs_space, act_space

    def _get_algorithm_trainer(self) -> BaseAlgorithmTrainer:
        algo_name = self._config.backend.algorithm.name.lower()
        if algo_name == "ppo":
            from l2l_lab.rllib.algorithms.ppo import PPOTrainer
            return PPOTrainer(self._config)
        elif algo_name == "impala":
            from l2l_lab.rllib.algorithms.impala import IMPALATrainer
            return IMPALATrainer(self._config)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: ppo, impala")

    def _get_policy_name(self) -> str:
        policy_cfg = getattr(self._config.backend.algorithm.config, "policy", None)
        if policy_cfg and policy_cfg.use_multiple_policies:
            return "main_policy"
        return "shared_policy"
