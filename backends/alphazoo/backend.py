from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

import torch

from backends.base import AlgorithmBackend, StepResult
from backends.obs_utils import make_wrapper, make_obs_to_state
from envs.registry import create_env

if TYPE_CHECKING:
    from agents.agent import Agent
    from configs.definition.training.TrainingConfig import TrainingConfig


def _get_shapes(env, obs_to_state):
    """Compute state shape and action space shape from a PettingZoo env."""
    import numpy as np

    obs = env.observe(env.agent_selection)
    state = obs_to_state(obs, None)
    state_shape = tuple(state.shape[1:])

    action_space = env.action_space(env.agent_selection)
    if hasattr(action_space, 'n'):
        num_actions = action_space.n
    elif hasattr(action_space, 'shape'):
        num_actions = int(np.prod(action_space.shape))
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")
    action_space_shape = (num_actions,)

    return state_shape, action_space_shape


class AlphaZooBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self._alphazoo = None
        self._config: Optional[TrainingConfig] = None
        self._train_thread: Optional[threading.Thread] = None
        self._obs_to_state = None

    @property
    def name(self) -> str:
        return "alphazoo"

    def _build_model(self, config: TrainingConfig, state_shape, action_space_shape):
        network_class = config.network.get_network_class()
        kwargs = config.network.to_kwargs()
        architecture = config.network.architecture

        if architecture in ("ResNet", "ConvNet"):
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

        model.recurrent = False
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
        self._obs_to_state = make_obs_to_state(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = _get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        print(f"\nEnvironment Info:")
        print(f"  State shape: {state_shape}")
        print(f"  Action space: {action_space_shape}")
        print()
        print("=" * 70)

        model = self._build_model(config, state_shape, action_space_shape)

        game = make_wrapper(env, config.network.architecture, env_config.obs_space_format)

        az_config = config.algorithm.config
        az_config.running.training_steps = config.algorithm.iterations

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=model,
        )

        print(f"\n✓ AlphaZoo instance created successfully!")

    def restore(self, config: TrainingConfig, model_dir: Path, checkpoint_dir: Path):
        from checkpoint_utils import get_checkpoint_path, load_checkpoint_data
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
        self._obs_to_state = make_obs_to_state(env_config.obs_space_format)

        env.reset()
        state_shape, action_space_shape = _get_shapes(env, self._obs_to_state)
        self._state_shape = state_shape
        self._action_space_shape = action_space_shape

        model = self._build_model(config, state_shape, action_space_shape)

        game = make_wrapper(env, config.network.architecture, env_config.obs_space_format)

        cp_path = get_checkpoint_path(model_dir, config.continue_from_iteration)
        optimizer_state_dict = None
        scheduler_state_dict = None
        replay_buffer_state = None
        if cp_path and cp_path.exists():
            cp_data_raw = torch.load(cp_path, weights_only=False)
            model.load_state_dict(cp_data_raw["model_state_dict"])
            optimizer_state_dict = cp_data_raw.get("optimizer_state_dict")
            scheduler_state_dict = cp_data_raw.get("scheduler_state_dict")
            replay_buffer_state = cp_data_raw.get("replay_buffer_state")
            print(f"✓ Model weights restored from {cp_path}")
        else:
            print("No checkpoint found. Starting fresh.")

        az_config = config.algorithm.config
        az_config.running.training_steps = config.algorithm.iterations

        self._alphazoo = AlphaZoo(
            env=game,
            config=az_config,
            model=model,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            replay_buffer_state=replay_buffer_state,
        )

        cp_data = load_checkpoint_data(model_dir, config.continue_from_iteration)
        start_iteration = cp_data.iteration if cp_data else 0

        return start_iteration, cp_data

    def start_training(self, start_iteration: int, total_iterations: int) -> None:
        az = self._alphazoo
        az.config.running.training_steps = total_iterations
        az.starting_step = start_iteration

        def _on_step_end(alphazoo_instance, step, raw_metrics):
            metrics = {
                "episode_len_mean": raw_metrics.get("episode_len_mean", 0),
                "policy_loss": raw_metrics.get("policy_loss"),
                "value_loss": raw_metrics.get("value_loss"),
                "combined_loss": raw_metrics.get("combined_loss"),
                "learning_rate": raw_metrics.get("learning_rate"),
                "replay_buffer_size": raw_metrics.get("replay_buffer_size"),
                "step_time": raw_metrics.get("step_time"),
            }
            self.metrics_queue.put(StepResult(iteration=step, metrics=metrics))

        def _train_loop():
            try:
                az.train(on_step_end=_on_step_end)
            except Exception as e:
                print(f"\n✗ AlphaZoo training error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.metrics_queue.put(None)

        self._train_thread = threading.Thread(target=_train_loop, daemon=True)
        self._train_thread.start()

    def create_agent_from_checkpoint(self, checkpoint_path: Path) -> Agent:
        from backends.alphazoo.agent import AlphaZooPolicyAgent
        from alphazoo import Network_Manager

        cp_data = torch.load(checkpoint_path, weights_only=False)

        model = self._build_model(self._config, self._state_shape, self._action_space_shape)
        model.load_state_dict(cp_data["model_state_dict"])
        model.eval()
        model.cpu()

        nm = Network_Manager.__new__(Network_Manager)
        nm.model = model
        nm.device = "cpu"
        return AlphaZooPolicyAgent(nm, self._obs_to_state, label="checkpoint")

    def create_eval_agent(self) -> Agent:
        from backends.alphazoo.agent import AlphaZooPolicyAgent
        from alphazoo import Network_Manager

        model_copy = deepcopy(self._alphazoo.latest_network.get_model())
        model_copy.eval()
        model_copy.cpu()
        nm = Network_Manager.__new__(Network_Manager)
        nm.model = model_copy
        nm.device = "cpu"

        return AlphaZooPolicyAgent(nm, self._obs_to_state, label="current")

    def get_weight_parameters(self) -> Optional[Iterator]:
        return self._alphazoo.latest_network.get_model().parameters()

    def save_checkpoint(self, checkpoint_dir: Path, iteration: int, metrics: Dict[str, List]) -> None:
        az = self._alphazoo

        checkpoint = {
            "model_state_dict": az.latest_network.get_model().state_dict(),
            "optimizer_state_dict": az.get_optimizer_state_dict(),
            "scheduler_state_dict": az.get_scheduler_state_dict(),
            "replay_buffer_state": az.get_replay_buffer_state(),
            "iteration": iteration,
            "metrics": metrics,
            "backend": self.name,
        }

        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

    def wait_for_completion(self) -> None:
        if self._train_thread is not None:
            self._train_thread.join()

    def shutdown(self) -> None:
        import ray
        if ray.is_initialized():
            ray.shutdown()
