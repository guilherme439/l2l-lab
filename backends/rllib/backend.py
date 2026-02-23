from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

import torch
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from backends.base import AlgorithmBackend, StepResult
from envs.registry import create_env

if TYPE_CHECKING:
    from agents.agent import Agent
    from configs.definition.training.TrainingConfig import TrainingConfig
    from rllib.algorithms.base import BaseAlgorithmTrainer


class RLlibBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self.algo: Any = None
        self.algo_trainer: Optional[BaseAlgorithmTrainer] = None
        self._train_thread: Optional[threading.Thread] = None
        self._config: Optional[TrainingConfig] = None

    @property
    def name(self) -> str:
        if self._config:
            return f"rllib_{self._config.algorithm.name}"
        return "rllib"

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
            from rllib.algorithms.ppo import PPOTrainer
            return PPOTrainer(self._config)
        elif algo_name == "impala":
            from rllib.algorithms.impala import IMPALATrainer
            return IMPALATrainer(self._config)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: ppo, impala")

    def _get_policy_name(self) -> str:
        policy_cfg = self._config.algorithm.config.policy
        if policy_cfg and policy_cfg.use_multiple_policies:
            return "main_policy"
        return "shared_policy"

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
        def _train_loop():
            try:
                for i in range(start_iteration, total_iterations):
                    result = self.algo.train()
                    metrics = self.algo_trainer.extract_metrics(result)
                    metrics["_rllib_result"] = result
                    self.metrics_queue.put(StepResult(iteration=i + 1, metrics=metrics))
            except Exception as e:
                print(f"\n✗ Training thread error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.metrics_queue.put(None)

        self._train_thread = threading.Thread(target=_train_loop, daemon=True)
        self._train_thread.start()

    def create_eval_agent(self) -> Agent:
        from agents.rl_module_agent import RLModuleAgent
        rl_module = self.algo.get_module(self._get_policy_name())
        rl_module.eval()
        agent = RLModuleAgent(rl_module, label="current")
        return agent

    def create_agent_from_checkpoint(self, checkpoint_path: Path) -> Agent:
        from agents.policy_agent import PolicyAgent
        from configs.definition.training.NetworkConfig import CONV_ARCHITECTURES, MLP_ARCHITECTURES

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config = checkpoint["model_config"]
        architecture = model_config.get("architecture", "ConvNet")
        obs_space_format = model_config.get("obs_space_format", "channels_first")
        network_class = model_config["network_class"]
        network_kwargs = model_config.get("network_kwargs", {})

        obs_space = checkpoint["observation_space"]
        act_space = checkpoint["action_space"]
        inner_obs_space = obs_space["observation"]

        if architecture in CONV_ARCHITECTURES:
            obs_shape = inner_obs_space.shape
            backbone = network_class(
                in_channels=obs_shape[0],
                policy_channels=act_space.n // (obs_shape[1] * obs_shape[2]),
                **network_kwargs,
            )
        elif architecture in MLP_ARCHITECTURES:
            backbone = network_class(out_features=act_space.n, **network_kwargs)
            dummy = torch.zeros((1,) + inner_obs_space.shape, dtype=torch.float32)
            with torch.no_grad():
                _ = backbone(dummy)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        backbone.eval()
        return PolicyAgent(backbone, obs_space_format, label="checkpoint")

    def _set_module_training(self) -> None:
        try:
            self.algo.get_module(self._get_policy_name()).train()
        except Exception:
            pass

    def get_weight_parameters(self) -> Optional[Iterator]:
        rl_module = self.algo.get_module(self._get_policy_name())
        return rl_module.parameters()

    def save_checkpoint(self, checkpoint_dir: Path, iteration: int, metrics: Dict[str, List]) -> None:
        rl_module = self.algo.get_module(self._get_policy_name())
        checkpoint = {
            "backbone_state_dict": rl_module.backbone.state_dict(),
            "observation_space": rl_module.observation_space,
            "action_space": rl_module.action_space,
            "model_config": rl_module.model_config,
            "iteration": iteration,
            "metrics": metrics,
            "backend": self.name,
        }

        model_path = checkpoint_dir / "checkpoint.pt"
        torch.save(checkpoint, model_path)

        # Also save model.cp for backward compat
        compat_path = checkpoint_dir / "model.cp"
        torch.save(checkpoint, compat_path)

        algo_path = checkpoint_dir / "algo_checkpoint"
        self.algo.save_to_path(str(algo_path.absolute()))

    def wait_for_completion(self) -> None:
        if self._train_thread is not None:
            self._train_thread.join()

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
