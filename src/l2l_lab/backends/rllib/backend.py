from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import torch
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from l2l_lab.backends.backend_base import AlgorithmBackend, StepResult
from l2l_lab.backends.checkpoint_writer import CheckpointWriter
from l2l_lab.envs.registry import create_env
from l2l_lab.utils.checkpoint import load_checkpoint_file, load_model_state_dict
from l2l_lab.utils.common import check_interval

if TYPE_CHECKING:
    from l2l_lab.configs.training.TrainingConfig import TrainingConfig
    from l2l_lab.rllib.algorithms.base import BaseAlgorithmTrainer


class _CheckpointWriter(CheckpointWriter):

    def __init__(self, backend: "RLlibBackend") -> None:
        self._backend = backend
        super().__init__()

    def write(self, snapshot: Dict[str, Any], path: Path) -> None:
        model_dir = path / "model"
        model_dir.mkdir(exist_ok=True)
        torch.save(snapshot["model_state_dict"], model_dir / "weights.cp")
        (model_dir / "base_class.pkl").write_bytes(snapshot["network_template_bytes"])
        self._backend.algo.save_to_path(str((path / "algo").absolute()))


class RLlibBackend(AlgorithmBackend):

    def __init__(self):
        super().__init__()
        self.algo: Any = None
        self.algo_trainer: Optional[BaseAlgorithmTrainer] = None
        self._config: Optional[TrainingConfig] = None
        self._input_shape: Optional[tuple] = None
        self._num_actions: Optional[int] = None
        self._network_template_bytes: Optional[bytes] = None
        self._writer = _CheckpointWriter(self)

    @property
    def name(self) -> str:
        if self._config:
            return f"rllib_{self._config.backend.algorithm.name}"
        return "rllib"

    def init(self) -> None:
        import ray
        if not ray.is_initialized():
            print()
            ray.init(
                ignore_reinit_error=True,
                _system_config={
                    "gcs_rpc_server_reconnect_timeout_s": 120,
                    "gcs_server_request_timeout_seconds": 120,
                    "local_fs_capacity_threshold": 0.99
                },
                object_store_memory=2 * 1024 * 1024 * 1024,
            )
            print()

    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
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

        print(f"\nBuilding {config.backend.algorithm.name.upper()} algorithm...\n")
        self.algo = rllib_config.build_algo()
        self.algo_trainer.algo = self.algo
        self._network_template_bytes = self._pickle_network(self._get_backbone())

        self._start_iteration = 0
        self._total_iterations = config.backend.algorithm.total_iterations

        print("\n✓ Algorithm built successfully!")

    def restore(self, config: TrainingConfig, model_dir: Path) -> int:
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

        loaded_iteration = self.algo_trainer.load_checkpoint_for_continue(
            rllib_config, model_dir, target_iteration=config.backend.continue_from_iteration
        )
        self.algo = self.algo_trainer.algo
        self._network_template_bytes = self._pickle_network(self._get_backbone())

        self._start_iteration = loaded_iteration + 1
        self._total_iterations = config.backend.algorithm.total_iterations

        return loaded_iteration

    def _train(self) -> None:
        info_interval = self._config.common.info_interval
        try:
            for step in range(self._start_iteration, self._total_iterations):
                if self._stop_event.is_set():
                    break
                result = self.algo.train()
                metrics = self.algo_trainer.extract_metrics(result)

                self._print_step_info(step, metrics)
                if check_interval(step, info_interval):
                    self._print_training_info(step, metrics)

                checkpoint_path: Optional[Path] = None
                if check_interval(step, self._checkpoint_interval):
                    print(f"\nSaving {self.name} checkpoint for iteration {step}")
                    snapshot = self._capture_snapshot()
                    checkpoint_path = self._checkpoint_base_dir / "checkpoints" / str(step)
                    checkpoint_path.mkdir(exist_ok=True)
                    self._writer.enqueue(snapshot, checkpoint_path)

                self.step_queue.put(StepResult(
                    iteration=step,
                    metrics=metrics,
                    checkpoint_path=checkpoint_path,
                ))
        finally:
            self.step_queue.put(None)

    def get_eval_model(self) -> torch.nn.Module:
        # this reads the model in a different thread while writes are happening,
        # it should be a race condition but it never crashed so...
        model_copy = deepcopy(self._get_backbone()).cpu() 
        model_copy.eval()
        return model_copy

    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> torch.nn.Module:
        model_dir = checkpoint_dir / "model"
        backbone = torch.load(model_dir / "base_class.pkl", weights_only=False)
        state_dict = load_checkpoint_file(model_dir / "weights.cp")
        load_model_state_dict(backbone, state_dict)
        backbone.eval()
        return backbone

    def get_weight_parameters(self) -> Optional[Iterator]:
        rl_module = self.algo.get_module(self._get_policy_name())
        return rl_module.parameters()

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

    def shutdown(self) -> None:
        self._writer.stop()
        if self.algo is not None:
            self.algo.stop()
        import ray
        if ray.is_initialized():
            ray.shutdown()

    def on_checkpoint_saved(self, model_dir: Path, iteration: int) -> None:
        self.algo_trainer.update_opponent_policies(model_dir, iteration)

    def get_reporter_csv_keys(self) -> List[str]:
        return [
            "episode_len_mean",
            "episode_reward_mean",
            "total_loss",
            "policy_loss",
            "vf_loss",
            "entropy",
            "kl_divergence",
            "vf_explained_var",
            "learning_rate",
        ]

    def _capture_snapshot(self) -> Dict[str, Any]:
        return {
            "model_state_dict": deepcopy(self._get_backbone().state_dict()),
            "network_template_bytes": self._network_template_bytes,
        }

    def _get_backbone(self) -> torch.nn.Module:
        return self.algo.get_module(self._get_policy_name()).backbone

    @staticmethod
    def _pickle_network(model: torch.nn.Module) -> bytes:
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.getvalue()

    def _print_step_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        ep_len = metrics.get("episode_len_mean", 0) or 0
        total = self._config.backend.algorithm.total_iterations
        print(f"\n{iteration}/{total} | EpLen: {ep_len:6.1f}\n")

    def _print_training_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        timesteps = metrics.get("timesteps_lifetime", 0)
        curr_lr = metrics.get("learning_rate")

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
