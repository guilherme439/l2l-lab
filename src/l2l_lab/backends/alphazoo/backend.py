from __future__ import annotations

import io
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import torch
from alphazoo import AlphaZooRecurrentNet, AlphaZooConfig
from gymnasium.spaces.utils import flatdim

from l2l_lab.backends.backend_base import AlgorithmBackend, StepResult
from l2l_lab.backends.checkpoint_writer import CheckpointWriter
from l2l_lab.backends.obs_utils import make_wrapper, obs_to_state_provider
from l2l_lab.configs.training.network import MLPNetConfig, SNNetConfig
from l2l_lab.envs.registry import create_env
from l2l_lab.neural_networks.utils.builders import build_network
from l2l_lab.utils.checkpoint import (get_checkpoint_dir, load_checkpoint_file,
                                      load_model_state_dict)
from l2l_lab.utils.common import check_interval
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from l2l_lab.configs.training.TrainingConfig import TrainingConfig


class _CheckpointWriter(CheckpointWriter):

    def __init__(self, backend: "AlphaZooBackend") -> None:
        self._backend = backend
        super().__init__()

    def write(self, snapshot: Dict[str, Any], path: Path) -> None:
        model_dir = path / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save(snapshot["model_state_dict"], model_dir / "weights.cp")
        (model_dir / "base_class.pkl").write_bytes(snapshot["network_template_bytes"])

        algo_dir = path / "algo"
        algo_dir.mkdir(exist_ok=True)
        torch.save(snapshot["algo"], algo_dir / "state.cp")


class AlphaZooBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self._alphazoo = None
        self._config: Optional[TrainingConfig] = None
        self._model = None
        self._obs_to_state = None
        self._network_template_bytes: Optional[bytes] = None
        self._writer = _CheckpointWriter(self)

    @property
    def name(self) -> str:
        return "alphazoo"

    def init(self) -> None:
        import ray
        if not ray.is_initialized():
            logger.info("")
            ray.init(
                ignore_reinit_error=True,
                _system_config={"local_fs_capacity_threshold": 0.99},
            )
            logger.info("")

    def shutdown(self) -> None:
        import ray

        self._writer.stop()
        if ray.is_initialized():
            ray.shutdown()

    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
        from alphazoo import AlphaZoo

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
        self._network_template_bytes = self._pickle_network(self._model)

        game = make_wrapper(env, env_config.obs_space_format)

        self._total_iterations = config.backend.algorithm.total_iterations

        az_config: AlphaZooConfig = config.backend.algorithm.config
        az_config.data.observation_format = env_config.obs_space_format
        az_config.data.network_input_format = "channels_first"
        az_config.running.training_steps = self._total_iterations

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=self._model,
        )

        logger.info(f"\n✓ AlphaZoo instance created successfully!")

    def restore(self, config: TrainingConfig, model_dir: Path) -> int:
        from alphazoo import AlphaZoo

        self._config = config
        env_config = config.env

        env = create_env(env_config.name, **env_config.kwargs)
        self._obs_to_state = obs_to_state_provider(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = self._get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        self._model = self._build_model(config, state_shape, action_space_shape)
        self._initialize_lazy_params(state_shape)
        self._network_template_bytes = self._pickle_network(self._model)

        game = make_wrapper(env, env_config.obs_space_format)

        backend_cfg = config.backend
        target_iteration = backend_cfg.continue_from_iteration

        checkpoint_dir = get_checkpoint_dir(model_dir, target_iteration)

        optimizer_state_dict = None
        scheduler_state_dict = None
        replay_buffer_state = None
        loaded_iteration = 0

        weights_path = checkpoint_dir / "model" / "weights.cp" if checkpoint_dir is not None else None
        if weights_path is not None and weights_path.exists():
            model_state_dict = load_checkpoint_file(weights_path)
            load_model_state_dict(self._model, model_state_dict)
            loaded_iteration = int(checkpoint_dir.name)
            logger.info(f"\n✓ Model weights restored from {weights_path}")

            algo_state_path = checkpoint_dir / "algo" / "state.cp"
            if algo_state_path.exists():
                algo_state = load_checkpoint_file(algo_state_path)
                if backend_cfg.load_optimizer:
                    optimizer_state_dict = algo_state.get("optimizer_state_dict")
                if backend_cfg.load_scheduler:
                    scheduler_state_dict = algo_state.get("scheduler_state_dict")
                replay_buffer_state = algo_state.get("replay_buffer_state")
        else:
            logger.info("No checkpoint found. Starting fresh.")

        self._start_iteration = loaded_iteration + 1
        self._total_iterations = config.backend.algorithm.total_iterations

        az_config: AlphaZooConfig = backend_cfg.algorithm.config
        az_config.data.observation_format = env_config.obs_space_format
        az_config.data.network_input_format = "channels_first"
        az_config.running.training_steps = self._total_iterations

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=self._model,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            replay_buffer_state=replay_buffer_state,
            start_iteration=self._start_iteration,
        )

        return loaded_iteration

    def get_reporter_csv_keys(self) -> List[str]:
        return [
            "episode_len_mean",
            "policy_loss",
            "value_loss",
            "combined_loss",
            "learning_rate",
        ]

    def get_eval_model(self) -> torch.nn.Module:
        model_copy = deepcopy(self._model).cpu()
        model_copy.eval()
        return model_copy

    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> torch.nn.Module:
        model_dir = checkpoint_dir / "model"
        model = torch.load(model_dir / "base_class.pkl", weights_only=False)
        state_dict = load_checkpoint_file(model_dir / "weights.cp")
        load_model_state_dict(model, state_dict)
        model.eval()
        return model

    def get_weight_parameters(self) -> Optional[Iterator]:
        return self._model.parameters()

    def save_final_checkpoint(self, iteration: int) -> Optional[Path]:
        if self._checkpoint_interval <= 0 or self._checkpoint_base_dir is None:
            return None
        snapshot = self._capture_snapshot()
        checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(iteration)
        checkpoint_path.mkdir(exist_ok=True)
        self._writer.write(snapshot, checkpoint_path)
        return checkpoint_path

    def wait_for_pending_checkpoints(self) -> None:
        self._writer.wait_for_idle()

    
    def _train(self) -> None:
        try:
            az = self._alphazoo

            info_interval = self._config.common.info_interval

            def _on_step_end(alphazoo_instance, step, metrics):
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
                self._print_step_info(step, public_metrics)
                if check_interval(step, info_interval):
                    self._print_training_info(step, public_metrics)

                checkpoint_path: Optional[Path] = None
                if check_interval(step, self._checkpoint_interval):
                    snapshot = self._capture_snapshot()
                    checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(step)
                    checkpoint_path.mkdir(exist_ok=True)
                    self._writer.enqueue(snapshot, checkpoint_path)

                eval_model = self.get_eval_model() if self._needs_snapshot(step) else None

                self.step_queue.put(StepResult(
                    iteration=step,
                    metrics=public_metrics,
                    checkpoint_path=checkpoint_path,
                    eval_model=eval_model,
                ))
                if self._stop_event.is_set():
                    return False

            az.train(on_step_end=_on_step_end)
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
    
    def _print_step_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        ep_len = metrics.get("episode_len_mean", 0) or 0
        total = self._config.backend.algorithm.total_iterations
        logger.info(f"\nIteration {iteration}/{total} finished | EpLen: {ep_len:6.1f}\n")
        
    def _capture_snapshot(self) -> Dict[str, Any]:
        az = self._alphazoo
        return {
            "model_state_dict": deepcopy(self._model.state_dict()),
            "network_template_bytes": self._network_template_bytes,
            "algo": {
                "optimizer_state_dict": deepcopy(az.get_optimizer_state_dict()),
                "scheduler_state_dict": deepcopy(az.get_scheduler_state_dict()),
                "replay_buffer_state": deepcopy(az.get_replay_buffer_state_dict()),
            },
        }

    def _pickle_network(self, model: torch.nn.Module) -> bytes:
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.getvalue()

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
