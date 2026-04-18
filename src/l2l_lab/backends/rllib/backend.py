from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

import torch
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from l2l_lab.backends.base import AlgorithmBackend, StepResult
from l2l_lab.configs.training.NetworkConfig import CONV_ARCHITECTURES, MLP_ARCHITECTURES
from l2l_lab.envs.registry import create_env

if TYPE_CHECKING:
    from l2l_lab.configs.training.TrainingConfig import TrainingConfig
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
    def name(self) -> str:
        if self._config:
            return f"rllib_{self._config.algorithm.name}"
        return "rllib"

    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
        import ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                _system_config={
                    "gcs_rpc_server_reconnect_timeout_s": 120,
                    "gcs_server_request_timeout_seconds": 120,
                    "local_fs_capacity_threshold": 0.99
                },
                object_store_memory=2 * 1024 * 1024 * 1024,
            )

        self._config = config
        self._register_env(config.env)
        obs_space, act_space = self._get_spaces(config.env)
        self._input_shape = obs_space["observation"].shape
        self._num_actions = act_space.n

        print(f"\nEnvironment Info:")
        print("  Observation space:")
        for key, space in obs_space.spaces.items():
            if hasattr(space, 'shape'):
                print(f"    {key}: shape={space.shape}, dtype={space.dtype}")
            else:
                print(f"    {key}: {space}")
        print(f"  Action space: {act_space}")
        print()
        print("=" * 70)

        self.algo_trainer = self._get_algorithm_trainer()
        rllib_config = self.algo_trainer.build_config(
            config.env.name, config.env.obs_space_format, obs_space, act_space
        )

        print(f"\nBuilding {config.algorithm.name.upper()} algorithm...\n")
        self.algo = rllib_config.build_algo()
        self.algo_trainer.algo = self.algo
        print("\n✓ Algorithm built successfully!")

    def restore(self, config: TrainingConfig, model_dir: Path, checkpoint_dir: Path) -> int:
        import ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                _system_config={
                    "gcs_rpc_server_reconnect_timeout_s": 120,
                    "gcs_server_request_timeout_seconds": 120,
                    "local_fs_capacity_threshold": 0.99
                },
                object_store_memory=2 * 1024 * 1024 * 1024,
            )

        self._config = config
        self._register_env(config.env)
        obs_space, act_space = self._get_spaces(config.env)
        self._input_shape = obs_space["observation"].shape
        self._num_actions = act_space.n

        print(f"\nEnvironment Info:")
        print("  Observation space:")
        for key, space in obs_space.spaces.items():
            if hasattr(space, 'shape'):
                print(f"    {key}: shape={space.shape}, dtype={space.dtype}")
            else:
                print(f"    {key}: {space}")
        print(f"  Action space: {act_space}")
        print()
        print("=" * 70)

        self.algo_trainer = self._get_algorithm_trainer()
        rllib_config = self.algo_trainer.build_config(
            config.env.name, config.env.obs_space_format, obs_space, act_space
        )

        start_iteration, cp_data = self.algo_trainer.load_checkpoint_for_continue(
            rllib_config, model_dir, target_iteration=config.continue_from_iteration
        )
        self.algo = self.algo_trainer.algo
        return start_iteration, cp_data

    def start_training(self, start_iteration: int, total_iterations: int) -> None:
        self._stop_event.clear()

        def _train():
            try:
                for i in range(start_iteration, total_iterations):
                    if self._stop_event.is_set():
                        break
                    result = self.algo.train()
                    metrics = self.algo_trainer.extract_metrics(result)
                    metrics["_rllib_result"] = result

                    self.step_queue.put(StepResult(
                        iteration=i + 1,
                        metrics=metrics,
                    ))
            finally:
                self.step_queue.put(None)

        self._training_thread = threading.Thread(target=_train, daemon=True)
        self._training_thread.start()

    def get_eval_model(self) -> torch.nn.Module:
        rl_module = self.algo.get_module(self._get_policy_name())
        backbone = rl_module.backbone
        backbone.eval()
        return backbone

    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> torch.nn.Module:
        cp = torch.load(checkpoint_dir / "training" / "data.pt", weights_only=False)
        cfg = self._config

        architecture = cfg.network.architecture
        network_class = cfg.network.get_network_class()
        network_kwargs = cfg.network.to_kwargs()

        if architecture in CONV_ARCHITECTURES:
            in_channels = self._input_shape[0]
            rows, cols = self._input_shape[1], self._input_shape[2]
            backbone = network_class(
                in_channels=in_channels,
                policy_channels=self._num_actions // (rows * cols),
                **network_kwargs,
            )
        elif architecture in MLP_ARCHITECTURES:
            backbone = network_class(out_features=self._num_actions, **network_kwargs)
            dummy = torch.zeros((1,) + self._input_shape, dtype=torch.float32)
            with torch.no_grad():
                _ = backbone(dummy)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        backbone.load_state_dict(cp["backbone_state_dict"])
        backbone.eval()
        return backbone

    def get_checkpoint_data(self) -> Dict[str, Any]:
        rl_module = self.algo.get_module(self._get_policy_name())
        return {
            "backbone_state_dict": deepcopy(rl_module.backbone.state_dict()),
        }

    def get_weight_parameters(self) -> Optional[Iterator]:
        rl_module = self.algo.get_module(self._get_policy_name())
        return rl_module.parameters()

    def save_checkpoint(self, checkpoint_dir: Path, iteration: int,
                        metrics: Dict[str, List], checkpoint_data: Dict[str, Any]) -> None:
        cfg = self._config

        model_dir = checkpoint_dir / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save({
            "state_dict": checkpoint_data["backbone_state_dict"],
            "architecture": cfg.network.architecture,
            "network_kwargs": cfg.network.to_kwargs(),
            "obs_space_format": cfg.env.obs_space_format,
            "input_shape": self._input_shape,
            "num_actions": self._num_actions,
        }, model_dir / "checkpoint.pt")

        training_dir = checkpoint_dir / "training"
        training_dir.mkdir(exist_ok=True)
        torch.save({
            "backbone_state_dict": checkpoint_data["backbone_state_dict"],
            "iteration": iteration,
            "metrics": metrics,
            "backend": self.name,
        }, training_dir / "data.pt")
        self.algo.save_to_path(str((training_dir / "algo_checkpoint").absolute()))

    def shutdown(self) -> None:
        if self.algo is not None:
            self.algo.stop()
        import ray
        if ray.is_initialized():
            ray.shutdown()

    def update_opponent_policies(self, model_dir: Path, checkpoint_iteration: int) -> None:
        if hasattr(self.algo_trainer, 'update_opponent_policies'):
            self.algo_trainer.update_opponent_policies(model_dir, checkpoint_iteration)

    def print_training_info(self, result: Dict[str, Any]) -> None:
        policy_name = self._get_policy_name()
        learner_info = result.get("learners", {}).get(policy_name, {})
        env_runners = result.get("env_runners", {})

        timesteps = env_runners.get("num_env_steps_sampled_lifetime", 0)
        curr_lr = learner_info.get("default_optimizer_learning_rate", None)

        print()
        print("  ┌─ Training Info ─────────────────────────────")
        print(f"  │ Timesteps: {timesteps:,}")
        if curr_lr is not None:
            print(f"  │ Learning Rate: {curr_lr:.2e}")
        print("  └──────────────────────────────────────────────")
        print()

    def _register_env(self, env_config) -> None:
        def env_creator(config: Dict):
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
        algo_name = self._config.algorithm.name.lower()
        if algo_name == "ppo":
            from l2l_lab.rllib.algorithms.ppo import PPOTrainer
            return PPOTrainer(self._config)
        elif algo_name == "impala":
            from l2l_lab.rllib.algorithms.impala import IMPALATrainer
            return IMPALATrainer(self._config)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: ppo, impala")

    def _get_policy_name(self) -> str:
        policy_cfg = self._config.algorithm.config.policy
        if policy_cfg and policy_cfg.use_multiple_policies:
            return "main_policy"
        return "shared_policy"
    
    def _set_module_training(self) -> None:
        try:
            self.algo.get_module(self._get_policy_name()).train()
        except Exception:
            pass
