from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

import torch
from gymnasium.spaces.utils import flatdim

from l2l_lab.backends.base import AlgorithmBackend, StepResult
from l2l_lab.backends.obs_utils import make_wrapper, obs_to_state_provider
from l2l_lab.envs.registry import create_env

if TYPE_CHECKING:
    from l2l_lab.configs.training.TrainingConfig import TrainingConfig


class _EarlyStopTrainingException(Exception):
    """Raised inside the AlphaZoo callback to break out of az.train()."""
    pass


class AlphaZooBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self._alphazoo = None
        self._config: Optional[TrainingConfig] = None
        self._model = None
        self._obs_to_state = None

    @property
    def name(self) -> str:
        return "alphazoo"

    def _build_model(self, config: TrainingConfig, state_shape, action_space_shape):
        network_class = config.network.get_network_class()
        kwargs = config.network.to_kwargs()
        architecture = config.network.architecture

        if architecture in ("ResNet", "ConvNet", "RecurrentNet"):
            in_channels = state_shape[0]
            rows, cols = state_shape[1], state_shape[2]
            num_actions = action_space_shape[0]
            policy_channels = num_actions // (rows * cols)
            model = network_class(
                in_channels=in_channels,
                policy_channels=policy_channels,
                **kwargs,
            )
        elif architecture == "MLPNet":
            out_features = action_space_shape[0]
            model = network_class(out_features=out_features, **kwargs)
            dummy = torch.zeros((1,) + state_shape, dtype=torch.float32)
            with torch.no_grad():
                _ = model(dummy)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return model

    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
        import ray
        from alphazoo import AlphaZoo

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                _system_config={"local_fs_capacity_threshold": 0.99}
            )

        self._config = config
        env_config = config.env

        env = create_env(env_config.name, **env_config.kwargs)
        self._obs_to_state = obs_to_state_provider(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = self._get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        print(f"\nEnvironment Info:")
        print(f"  State shape: {state_shape}")
        print(f"  Action space: {action_space_shape}")
        print()
        print("=" * 70)

        self._model = self._build_model(config, state_shape, action_space_shape)

        game = make_wrapper(env, env_config.obs_space_format)

        az_config = config.algorithm.config
        az_config.running.training_steps = config.algorithm.iterations
        az_config.data.observation_format = env_config.obs_space_format
        az_config.data.network_input_format = "channels_first"

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=self._model,
        )

        print(f"\n✓ AlphaZoo instance created successfully!")

    def restore(self, config: TrainingConfig, model_dir: Path, checkpoint_dir: Path):
        from l2l_lab.utils.checkpoint import get_checkpoint_path, load_checkpoint_data
        import ray
        from alphazoo import AlphaZoo

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                _system_config={"local_fs_capacity_threshold": 0.99}
            )

        self._config = config
        env_config = config.env

        env = create_env(env_config.name, **env_config.kwargs)
        self._obs_to_state = obs_to_state_provider(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = self._get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        self._model = self._build_model(config, state_shape, action_space_shape)

        game = make_wrapper(env, env_config.obs_space_format)

        cp_path = get_checkpoint_path(model_dir, config.continue_from_iteration)
        optimizer_state_dict = None
        scheduler_state_dict = None
        replay_buffer_state = None
        if cp_path and cp_path.exists():
            cp_data_raw = torch.load(cp_path, weights_only=False)
            self._model.load_state_dict(cp_data_raw["model_state_dict"])
            optimizer_state_dict = cp_data_raw.get("optimizer_state_dict")
            scheduler_state_dict = cp_data_raw.get("scheduler_state_dict")
            replay_buffer_state = cp_data_raw.get("replay_buffer_state")
            print(f"✓ Model weights restored from {cp_path}")
        else:
            print("No checkpoint found. Starting fresh.")

        az_config = config.algorithm.config
        az_config.running.training_steps = config.algorithm.iterations
        az_config.data.observation_format = env_config.obs_space_format
        az_config.data.network_input_format = "channels_first"

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=self._model,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            replay_buffer_state=replay_buffer_state,
        )

        cp_data = load_checkpoint_data(model_dir, config.continue_from_iteration)
        start_iteration = cp_data.iteration if cp_data else 0

        return start_iteration, cp_data

    def start_training(self, start_iteration: int, total_iterations: int) -> None:
        self._stop_event.clear()

        def _train():
            try:
                az = self._alphazoo
                az.config.running.training_steps = total_iterations
                az.starting_step = start_iteration

                def _on_step_end(alphazoo_instance, step, metrics):
                    self.step_queue.put(StepResult(
                        iteration=step,
                        metrics={
                            "episode_len_mean": metrics.get("rollout/episode_len_mean", 0),
                            "policy_loss": metrics.get("train/policy_loss"),
                            "value_loss": metrics.get("train/value_loss"),
                            "combined_loss": metrics.get("train/combined_loss"),
                            "learning_rate": metrics.get("train/learning_rate"),
                            "replay_buffer_size": metrics.get("train/replay_buffer_size"),
                        },
                    ))
                    if self._stop_event.is_set():
                        raise _EarlyStopTrainingException()

                az.train(on_step_end=_on_step_end)
            except _EarlyStopTrainingException:
                pass
            finally:
                self.step_queue.put(None)

        self._training_thread = threading.Thread(target=_train, daemon=True)
        self._training_thread.start()

    def get_eval_model(self) -> torch.nn.Module:
        model_copy = deepcopy(self._model)
        model_copy.eval()
        return model_copy

    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> torch.nn.Module:
        cp = torch.load(checkpoint_dir / "training" / "checkpoint.pt", weights_only=False)
        model = self._build_model(self._config, self._state_shape, self._action_space_shape)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()
        return model

    def get_checkpoint_data(self) -> Dict[str, Any]:
        az = self._alphazoo
        return {
            "model_state_dict": deepcopy(self._model.state_dict()),
            "optimizer_state_dict": deepcopy(az.get_optimizer_state_dict()),
            "scheduler_state_dict": deepcopy(az.get_scheduler_state_dict()),
            "replay_buffer_state": az.get_replay_buffer_state(),
        }

    def get_weight_parameters(self) -> Optional[Iterator]:
        return self._model.parameters()

    def save_checkpoint(self, checkpoint_dir: Path, iteration: int,
                        metrics: Dict[str, List], checkpoint_data: Dict[str, Any]) -> None:
        cfg = self._config

        model_dir = checkpoint_dir / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save({
            "state_dict": checkpoint_data["model_state_dict"],
            "architecture": cfg.network.architecture,
            "network_kwargs": cfg.network.to_kwargs(),
            "obs_space_format": cfg.env.obs_space_format,
            "input_shape": self._state_shape,
            "num_actions": self._action_space_shape[0],
        }, model_dir / "checkpoint.pt")

        training_dir = checkpoint_dir / "training"
        training_dir.mkdir(exist_ok=True)
        torch.save({
            **checkpoint_data,
            "iteration": iteration,
            "metrics": metrics,
            "backend": self.name,
        }, training_dir / "checkpoint.pt")

    def shutdown(self) -> None:
        import ray
        if ray.is_initialized():
            ray.shutdown()

    def _get_shapes(self, env, obs_to_state):
        """Compute state shape and action space shape from a PettingZoo env."""
        obs = env.observe(env.agent_selection)
        state = obs_to_state(obs, None)
        state_shape = tuple(state.shape[1:])

        action_space = env.action_space(env.agent_selection)
        num_actions = flatdim(action_space)
        action_space_shape = (num_actions,)

        return state_shape, action_space_shape
