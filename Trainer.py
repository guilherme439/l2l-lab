from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import torch
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from configs.definition.training.TrainingConfig import TrainingConfig
from envs.registry import create_env
from Tester import Tester
import graphs

if TYPE_CHECKING:
    from rllib.algorithms.base import BaseAlgorithmTrainer


MODELS_DIR = Path("models")


class Trainer:
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = TrainingConfig.from_yaml(config_path)
        self.algo = None
        self.metrics: Dict[str, List] = {}
        self.current_model_dir: Optional[Path] = None
        self._register_env()
    
    def _register_env(self) -> None:
        env_config = self.config.env
        
        def env_creator(config: Dict):
            env = create_env(env_config.name, **env_config.kwargs)
            return PettingZooEnv(env)
        
        register_env(env_config.name, env_creator)
    
    def _get_spaces(self):
        env_config = self.config.env
        env = create_env(env_config.name, **env_config.kwargs)
        wrapped = PettingZooEnv(env)
        first_agent = list(wrapped.observation_space.keys())[0]
        obs_space = wrapped.observation_space[first_agent]
        act_space = wrapped.action_space[first_agent]
        return obs_space, act_space
    
    def _setup_model_dir(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{self.config.name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        self.current_model_dir = model_dir
        return model_dir
    
    def _get_algorithm_trainer(self) -> BaseAlgorithmTrainer:
        algo_name = self.config.algorithm.name.lower()
        
        if algo_name == "ppo":
            from rllib.algorithms.ppo import PPOTrainer
            return PPOTrainer(self)
        elif algo_name == "impala":
            from rllib.algorithms.impala import IMPALATrainer
            return IMPALATrainer(self)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: ppo, impala")
    
    def _print_observation_space(self, obs_space) -> None:
        print("  Observation space:")
        for key, space in obs_space.spaces.items():
            if hasattr(space, 'shape'):
                print(f"    {key}: shape={space.shape}, dtype={space.dtype}")
            else:
                print(f"    {key}: {space}")
    
    def _init_metrics(self) -> None:
        self.metrics = {
            "iteration": [],
            "wins_vs_random": [],
            "losses_vs_random": [],
            "draws_vs_random": [],
        }
        if self.config.eval_vs_previous:
            self.metrics["wins_vs_previous"] = []
            self.metrics["losses_vs_previous"] = []
            self.metrics["draws_vs_previous"] = []
    
    def _record_eval(self, results, prefix: str) -> str:
        w_key, l_key, d_key = f"wins_{prefix}", f"losses_{prefix}", f"draws_{prefix}"
        
        if w_key in self.metrics:
            self.metrics[w_key].append(results.wins if results else None)
            self.metrics[l_key].append(results.losses if results else None)
            self.metrics[d_key].append(results.draws if results else None)
        
        if results:
            return f" | {prefix}: W{results.wins}/L{results.losses}/D{results.draws}"
        return ""
    
    def _print_training_info(self, result: Dict[str, Any]) -> None:
        learner_info = result.get("learners", {}).get("shared_policy", {})
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
    
    def train(self) -> None:
        cfg = self.config
        algo_cfg = cfg.algorithm
        env_cfg = cfg.env
        
        print("\n" * 3)
        print("=" * 70)
        print(f"\nTraining {cfg.env.name.upper()} with {algo_cfg.name.upper()}\n")
        print(f"  Name: {cfg.name}")
        print()
        print("=" * 70)
        
        self._setup_model_dir()
        obs_space, act_space = self._get_spaces()

        
        print(f"\nEnvironment Info:")
        self._print_observation_space(obs_space)
        print(f"  Action space: {act_space}")
        print()
        print("=" * 70)
        
        algo_trainer = self._get_algorithm_trainer()
        rllib_config = algo_trainer.build_config(env_cfg.name, env_cfg.obs_space_format, obs_space, act_space)
        
        if cfg.continue_training:
            start_iteration, cp_data = algo_trainer.load_checkpoint_for_continue(
                rllib_config, self.current_model_dir, target_iteration=cfg.continue_from_iteration
            )
            self.algo = algo_trainer.algo
            if cp_data and cp_data.metrics:
                self.metrics = cp_data.metrics
                print(f"✓ Loaded {len(self.metrics.get('iteration', []))} iterations of metrics from checkpoint")
            else:
                self._init_metrics()
        else:
            start_iteration = 0
            print(f"\nBuilding {algo_cfg.name.upper()} algorithm...\n")
            self.algo = rllib_config.build_algo()
            algo_trainer.algo = self.algo
            print("\n✓ Algorithm built successfully!")
            self._init_metrics()
        
        previous_checkpoint: Optional[Path] = None
        metrics_initialized = len(self.metrics.get("iteration", [])) > 0
        
        remaining_iterations = algo_cfg.iterations - start_iteration
        if remaining_iterations <= 0:
            print(f"\nAlready completed {start_iteration} iterations (target: {algo_cfg.iterations}). Nothing to do.")
            return

        print()
        print("=" * 70)
        print(f"\n\nStarting training for {remaining_iterations} iterations (from {start_iteration} to {algo_cfg.iterations})...")
        print("-" * 70)
        
        for i in range(start_iteration, algo_cfg.iterations):
            result = self.algo.train()
            metrics = algo_trainer.extract_metrics(result)
            
            if not metrics_initialized:
                for key in metrics.keys():
                    self.metrics[key] = []
                metrics_initialized = True

            self.metrics["iteration"].append(i + 1)
            for key, value in metrics.items():
                self.metrics[key].append(value)
            
            eval_str = ""
            
            results_random = None
            if (i + 1) % cfg.eval_interval == 0:
                results_random = Tester.evaluate_vs_random(self.algo, cfg.env, num_games=cfg.eval_games)
            eval_str += self._record_eval(results_random, "vs_random")
            
            results_prev = None
            if (i + 1) % cfg.checkpoint_interval == 0:
                if cfg.eval_vs_previous and previous_checkpoint is not None:
                    results_prev = Tester.evaluate_vs_checkpoint(
                        self.algo, previous_checkpoint, cfg.env, num_games=cfg.eval_games
                    )
                checkpoint_dir = self.save_checkpoint(i + 1)
                previous_checkpoint = checkpoint_dir / "model.cp"
            eval_str += self._record_eval(results_prev, "vs_previous")
            
            print(f"{i+1:8d}/{algo_cfg.iterations} | EpLen: {metrics['episode_len_mean']:6.1f}{eval_str}")
            
            if (i + 1) % cfg.plot_interval == 0:
                self.plot_progress()
            
            if (i + 1) % cfg.info_interval == 0:
                self._print_training_info(result)
        else:
            print("-" * 70)
            print(f"✓ {algo_cfg.name.upper()} Training completed!")
            
            self.save_checkpoint(algo_cfg.iterations)
            self.algo.stop()
            self.plot_progress()
    
    def save_checkpoint(self, iteration: int) -> Path:
        if self.algo is None:
            raise RuntimeError("No algorithm to save. Train first.")
        
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")
        
        checkpoint_dir = self.current_model_dir / "checkpoints" / str(iteration)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        rl_module = self.algo.get_module("shared_policy")
        checkpoint = {
            "backbone_state_dict": rl_module.backbone.state_dict(),
            "observation_space": rl_module.observation_space,
            "action_space": rl_module.action_space,
            "model_config": rl_module.model_config,
            "iteration": iteration,
            "metrics": self.metrics,
        }
        
        model_path = checkpoint_dir / "model.cp"
        torch.save(checkpoint, model_path)
        
        algo_path = checkpoint_dir / "algo_checkpoint"
        self.algo.save_to_path(str(algo_path.absolute()))
        
        print(f"\n  [Checkpoint saved: iter {iteration}]\n")
        return checkpoint_dir
    
    def get_latest_checkpoint(self, model_dir: Path) -> Optional[Path]:
        from checkpoint_utils import get_latest_checkpoint_dir
        return get_latest_checkpoint_dir(model_dir)
    
    def load_model(self, model_name: str, iteration: Optional[int] = None) -> None:
        from checkpoint_utils import get_algo_checkpoint_path
        
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{model_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        algo_checkpoint_path = get_algo_checkpoint_path(model_dir, iteration)
        if algo_checkpoint_path is None or not algo_checkpoint_path.exists():
            raise FileNotFoundError(f"No algo checkpoint found in: {model_dir}")
        
        self.current_model_dir = model_dir
        algo_trainer = self._get_algorithm_trainer()
        self.algo = algo_trainer.load_from_checkpoint(algo_checkpoint_path)
        print(f"✓ Algorithm loaded from: {algo_checkpoint_path}")
    
    def load_backbone_weights(self, model_name: str, iteration: Optional[int] = None) -> Dict[str, Any]:
        from checkpoint_utils import get_checkpoint_path, get_latest_checkpoint_path
        
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{model_name}"
        if iteration is not None:
            checkpoint_path = get_checkpoint_path(model_dir, iteration)
        else:
            checkpoint_path = get_latest_checkpoint_path(model_dir)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found in: {model_dir}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"✓ Backbone weights loaded from: {checkpoint_path}")
        return checkpoint

    
    def plot_progress(self) -> None:
        if not self.metrics.get("iteration"):
            print("No metrics to plot!")
            return
        
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")
        
        graphs_dir = self.current_model_dir / "graphs"
        graphs.plot_metrics(graphs_dir, self.metrics, self.config.eval_graph_split)
        print(f"\n📊 Graphs saved to: {graphs_dir}\n")
