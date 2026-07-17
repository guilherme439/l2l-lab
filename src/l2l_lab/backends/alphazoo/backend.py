import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, override

import torch
from alphazoo import AlphaZoo, AlphaZooRecurrentNet, AlphaZooConfig
from torch import nn
from gymnasium.spaces.utils import flatdim

from l2l_lab._utils.checkpoint import CheckpointUtils
from l2l_lab._utils.common import CommonUtils
from l2l_lab.backends.backend_base import AlgorithmBackend, StepResult
from l2l_lab.backends.obs_utils import make_wrapper, obs_to_state_provider
from l2l_lab.configs.training.network import MLPNetConfig, SNNetConfig
from l2l_lab.envs.registry import create_env
from l2l_lab.neural_networks.utils.builders import build_network
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from l2l_lab.configs.training.training_config import TrainingConfig


class AlphaZooBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self._alphazoo = None
        self._config: Optional[TrainingConfig] = None
        self._model = None
        self._obs_to_state = None
        self._game = None
        self._az_config: Optional[AlphaZooConfig] = None

    @property
    @override
    def name(self) -> str:
        return "alphazoo"

    @override
    def init(self) -> None:
        import ray
        if not ray.is_initialized():
            logger.info("")
            ray.init(
                ignore_reinit_error=True,
                _system_config={"local_fs_capacity_threshold": 0.99},
            )
            logger.info("")

    @override
    def shutdown(self) -> None:
        import ray

        if ray.is_initialized():
            ray.shutdown()

    @override
    def prepare(self, config: TrainingConfig) -> None:
        self._config = config
        env_config = config.env

        env = create_env(env_config.name, **env_config.kwargs)
        self._obs_to_state = obs_to_state_provider(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = self._get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        logger.info(f"\nEnvironment Info:")
        logger.info(f"  State shape: {state_shape}")
        logger.info(f"  Action space: {action_space_shape}")
        logger.info("")
        logger.info("=" * 70)

        self._model = self._build_model(config, state_shape, action_space_shape)
        self._initialize_lazy_params(state_shape)

        self._game = make_wrapper(env, env_config.obs_space_format)

        self._total_iterations = config.backend.algorithm.total_iterations

        az_config: AlphaZooConfig = config.backend.algorithm.config
        az_config.data.observation_format = env_config.obs_space_format
        az_config.data.network_input_format = "channels_first"
        az_config.running.training_steps = self._total_iterations
        self._az_config = az_config

    @override
    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        weights_path = checkpoint_dir / "weights.pt"
        model_state_dict = CheckpointUtils.load_checkpoint_file(weights_path)
        CheckpointUtils.load_model_state_dict(self._model, model_state_dict)
        logger.info(f"\n✓ Model weights restored from {weights_path}")

        backend_cfg = self._config.backend
        self._alphazoo = AlphaZoo.from_checkpoint(
            checkpoint_dir / "algo",
            env=self._game,
            config=self._az_config,
            model=self._model,
            load_model=False,
            load_optimizer=backend_cfg.load_optimizer,
            load_scheduler=backend_cfg.load_scheduler,
            load_replay_buffer=backend_cfg.load_replay_buffer,
        )

    @override
    def init_fresh(self) -> None:
        self._alphazoo = AlphaZoo(env=self._game, config=self._az_config, model=self._model)
        logger.info(f"\n✓ AlphaZoo instance created successfully!")

    @override
    def _get_live_model(self) -> nn.Module:
        return self._model

    @override
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> nn.Module:
        model = torch.load(CheckpointUtils.get_network_template_path(checkpoint_dir), weights_only=False)
        state_dict = CheckpointUtils.load_checkpoint_file(checkpoint_dir / "weights.pt")
        CheckpointUtils.load_model_state_dict(model, state_dict)
        model.eval()
        return model

    @override
    def get_weight_parameters(self) -> Optional[Iterator]:
        return self._model.parameters()

    @override
    def save_final_checkpoint(self, iteration: int) -> Optional[Path]:
        if self._checkpoint_interval <= 0 or self._checkpoint_base_dir is None:
            return None
        checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(iteration)
        # An abort can leave alphazoo's own step counter mid-step, past `iteration`,
        # so the final save stamps the iteration explicitly.
        self._write_checkpoint(checkpoint_path, iteration)
        return checkpoint_path

    @override
    def train(self) -> None:
        try:
            az = self._alphazoo

            info_interval = self._config.common.info_interval

            def _on_step_end(alphazoo_instance, current_iteration, metrics):
                iterations_completed = current_iteration + 1
                public_metrics = {
                    "episode_len_mean": metrics.get("rollout/episode_len_mean", 0),
                    "policy_loss": metrics.get("train/policy_loss"),
                    "value_loss": metrics.get("train/value_loss"),
                    "combined_loss": metrics.get("train/combined_loss"),
                    "learning_rate": metrics.get("train/learning_rate"),
                    "replay_buffer_size": metrics.get("train/replay_buffer_size"),
                    "cache_hit_ratio": metrics.get("inference/cache_hit_ratio"),
                    "cycle_size": metrics.get("inference/cycle_size"),
                    "batch_size": metrics.get("inference/batch_size"),
                }
                self.print_step_info(current_iteration, public_metrics)
                if CommonUtils.check_interval(iterations_completed, info_interval):
                    self.print_training_info(current_iteration, public_metrics)

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
                    metrics=public_metrics,
                    checkpoint_path=checkpoint_path,
                    eval_model=eval_model,
                ))
                if self._stop_event.is_set():
                    return False

            def _on_heartbeat(alphazoo_instance):
                return not self._stop_event.is_set()

            az.train(on_step_end=_on_step_end, on_heartbeat=_on_heartbeat)
        finally:
            self.step_queue.put(None)

    def _build_model(self, config: TrainingConfig, state_shape, action_space_shape):
        num_actions = action_space_shape[0]
        config.network.validate_for_env(state_shape, num_actions)

        if isinstance(config.network, (MLPNetConfig, SNNetConfig)):
            input_features = int(math.prod(state_shape))
            return build_network(
                config.network,
                input_features=input_features,
                num_actions=num_actions,
            )
        in_channels = state_shape[0]
        return build_network(
            config.network,
            in_channels=in_channels,
            num_actions=num_actions,
        )
    
    @override
    def print_step_info(self, iteration: int, metrics: dict[str, Any]) -> None:
        ep_len = metrics.get("episode_len_mean", 0) or 0
        total = self._config.backend.algorithm.total_iterations
        logger.info(f"\nIteration {iteration}/{total} finished | EpLen: {ep_len:6.1f}\n")
        
    def _write_checkpoint(self, path: Path, iteration: Optional[int] = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        az = self._alphazoo
        az.save(path / "algo", save_model=False, iteration=iteration)
        CheckpointUtils.atomic_write(path / "weights.pt", lambda temp_path: torch.save(az.get_model_state_dict(), temp_path))

    def _initialize_lazy_params(self, state_shape) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, *state_shape)
            if isinstance(self._model, AlphaZooRecurrentNet):
                self._model(dummy, iters_to_do=1)
            else:
                self._model(dummy)

    def _get_shapes(self, env, obs_to_state):
        """Compute state shape and action space shape from a PettingZoo env."""
        obs = env.observe(env.agent_selection)
        state = obs_to_state(obs, None)
        state_shape = tuple(state.shape[1:])

        action_space = env.action_space(env.agent_selection)
        num_actions = flatdim(action_space)
        action_space_shape = (num_actions,)

        return state_shape, action_space_shape
