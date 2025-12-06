from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from rl_scs.SCS_Game import SCS_Game

from configs.definition.TrainingConfig import TrainingConfig

from .DualHeadRLModule import DualHeadRLModule
from .SCSTester import SCSTester


MODELS_DIR = Path("models")
ENV_NAME = "scs_game"


class SCSTrainer:
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = TrainingConfig.from_yaml(config_path)
        self.algo = None
        self.metrics: Dict[str, List] = {}
        self.current_model_dir: Optional[Path] = None
        self._register_env()
    
    def _register_env(self) -> None:
        game_config = self.config.game_config
        debug = self.config.debug
        
        def env_creator(config: Dict):
            env = SCS_Game(
                game_config,
                action_mask_location="obs",
                obs_space_format="channels_first",
                debug=debug,
            )
            env.simulation_mode = False
            return PettingZooEnv(env)
        
        register_env(ENV_NAME, env_creator)
    
    def _get_spaces(self):
        env = SCS_Game(
            self.config.game_config,
            action_mask_location="obs",
            obs_space_format="channels_first",
        )
        wrapped = PettingZooEnv(env)
        obs_space = wrapped.observation_space["player_0"]
        act_space = wrapped.action_space["player_0"]
        return obs_space, act_space
    
    def _setup_model_dir(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{self.config.name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        self.current_model_dir = model_dir
        return model_dir
    
    def _get_network_class(self):
        return self.config.network.get_network_class()
    
    def _print_observation_space(self, obs_space) -> None:
        print("  Observation space:")
        for key, space in obs_space.spaces.items():
            if hasattr(space, 'shape'):
                print(f"    {key}: shape={space.shape}, dtype={space.dtype}")
            else:
                print(f"    {key}: {space}")
    
    def _load_checkpoint_for_continue(self, config) -> int:        
        algo_checkpoint_path = self.current_model_dir / "algo_checkpoint"
        if not algo_checkpoint_path.exists():
            print("\nNo existing checkpoint found. Starting fresh training...")
            print("\nBuilding PPO algorithm...\n")
            self.algo = config.build_algo()
            print("\n✓ Algorithm built successfully!")
            return 0
        
        print("\nLoading algorithm from checkpoint for continued training...")
        self.algo = PPO.from_checkpoint(str(algo_checkpoint_path.absolute()))
        
        start_iteration = 0
        latest_cp = self.get_latest_checkpoint(self.current_model_dir)
        if latest_cp:
            cp_data = torch.load(latest_cp, weights_only=False)
            start_iteration = cp_data.get("iteration", 0)
        else:
            model_cp = self.current_model_dir / "model.cp"
            if model_cp.exists():
                cp_data = torch.load(model_cp, weights_only=False)
                start_iteration = cp_data.get("iteration", 0)
        
        print(f"✓ Resuming from iteration {start_iteration}")
        return start_iteration
    
    def train(self) -> None:
        algo_cfg = self.config.algorithm
        if algo_cfg.name != "ppo":
            raise ValueError(f"Unsupported algorithm: {algo_cfg.name}")
        self._train_ppo()
    
    def _train_ppo(self) -> None:
        cfg = self.config
        algo = cfg.algorithm
        algo_cfg = algo.config
        
        print("\n" * 3)
        print("=" * 70)
        print(f"\nTraining {ENV_NAME.upper()} with {algo.name.upper()}\n")
        print(f"  Name: {cfg.name}")
        print(f"  Debug: {cfg.debug}")
        print()
        print("=" * 70)
        
        self._setup_model_dir()
        obs_space, act_space = self._get_spaces()
        
        print(f"\nEnvironment Info:")
        self._print_observation_space(obs_space)
        print(f"  Action space: {act_space}")
        print()
        print("=" * 70)
        
        network_class = self.config.network.get_network_class()
        
        config = (
            PPOConfig()
            .environment(
                env=ENV_NAME,
                disable_env_checking=True,
            )
            .env_runners(
                num_env_runners=algo_cfg["num_env_runners"],
                rollout_fragment_length=algo_cfg["rollout_fragment_length"],
            )
            .training(
                train_batch_size_per_learner=algo_cfg["train_batch_size_per_learner"],
                minibatch_size=algo_cfg["minibatch_size"],
                lr=algo_cfg["lr"],
                gamma=algo_cfg["gamma"],
                entropy_coeff=algo_cfg["entropy_coeff"],
            )
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "shared_policy": RLModuleSpec(
                            module_class=DualHeadRLModule,
                            observation_space=obs_space,
                            action_space=act_space,
                            model_config={
                                "network_class": network_class,
                                "network_kwargs": cfg.network.to_kwargs(),
                            },
                        ),
                    },
                ),
            )
            .framework("torch")
            .resources(num_gpus=0)
            .debugging(log_level="DEBUG" if cfg.debug else "WARN")
        )
              
        if cfg.continue_training:
            start_iteration = self._load_checkpoint_for_continue(config)
        else:
            start_iteration = 0
            print("\nBuilding PPO algorithm...\n")
            self.algo = config.build_algo()
            print("\n✓ Algorithm built successfully!")
        
        self.metrics = {
            "iteration": [],
            "episode_len_mean": [],
            "total_loss": [],
            "policy_loss": [],
            "vf_loss": [],
            "policy_entropy": [],
            "kl_divergence": [],
            "vf_explained_var": [],
            "win_rate_vs_random": [],
        }
        
        remaining_iterations = algo.iterations - start_iteration
        if remaining_iterations <= 0:
            print(f"\nAlready completed {start_iteration} iterations (target: {algo.iterations}). Nothing to do.")
            return

        print()
        print("=" * 70)
        print(f"\n\nStarting training for {remaining_iterations} iterations (from {start_iteration} to {algo.iterations})...")
        print("-" * 70)
        
        for i in range(start_iteration, algo.iterations):
            result = self.algo.train()
            
            env_runners = result.get("env_runners", {})
            episode_len_mean = env_runners.get("episode_len_mean", 0) or 0
            learner_stats = result.get("learners", {}).get("shared_policy", {})
            total_loss = learner_stats.get("total_loss", 0) or 0
            policy_loss = learner_stats.get("policy_loss", 0) or 0
            vf_loss = learner_stats.get("vf_loss", 0) or 0
            policy_entropy = learner_stats.get("entropy")
            kl_divergence = learner_stats.get("mean_kl_loss")
            vf_explained_var = learner_stats.get("vf_explained_var")

            win_rate = None
            if (i + 1) % cfg.eval_interval == 0:
                win_rate = SCSTester.evaluate_vs_random(self.algo, cfg.game_config, num_games=cfg.eval_games)
            
            self.metrics["iteration"].append(i + 1)
            self.metrics["episode_len_mean"].append(episode_len_mean)
            self.metrics["total_loss"].append(total_loss)
            self.metrics["policy_loss"].append(policy_loss)
            self.metrics["vf_loss"].append(vf_loss)
            self.metrics["policy_entropy"].append(policy_entropy)
            self.metrics["kl_divergence"].append(kl_divergence)
            self.metrics["vf_explained_var"].append(vf_explained_var)
            self.metrics["win_rate_vs_random"].append(win_rate)
            
            wr_str = f" => WinRate: {win_rate:.1%}" if win_rate is not None else ""
            print(f"{i+1:8d}/{algo.iterations} | EpLen: {episode_len_mean:6.1f} {wr_str}")
            
            if (i + 1) % cfg.checkpoint_interval == 0:
                self.save_checkpoint(i + 1)
            
            if (i + 1) % cfg.plot_interval == 0:
                self.plot_progress()
        
        print("-" * 70)
        print("✓ PPO Training completed!")
        
        self.save_model(algo.iterations)
        self.algo.stop()
        self.plot_progress()
    
    def save_checkpoint(self, iteration: int) -> Path:
        if self.algo is None:
            raise RuntimeError("No algorithm to save. Train first.")
        
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")
        
        checkpoints_dir = self.current_model_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        rl_module = self.algo.get_module("shared_policy")
        checkpoint = {
            "backbone_state_dict": rl_module.backbone.state_dict(),
            "observation_space": rl_module.observation_space,
            "action_space": rl_module.action_space,
            "model_config": rl_module.model_config,
            "iteration": iteration,
        }
        
        checkpoint_path = checkpoints_dir / f"model_iter_{iteration}.cp"
        torch.save(checkpoint, checkpoint_path)
        print(f"\n  [Checkpoint saved: iter {iteration}]\n")
        return checkpoint_path
    
    def save_model(self, iteration: int = -1) -> Path:
        if self.algo is None:
            raise RuntimeError("No algorithm to save. Train first.")
        
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")
        
        rl_module = self.algo.get_module("shared_policy")
        checkpoint = {
            "backbone_state_dict": rl_module.backbone.state_dict(),
            "observation_space": rl_module.observation_space,
            "action_space": rl_module.action_space,
            "model_config": rl_module.model_config,
            "iteration": iteration,
        }
        
        torch.save(checkpoint, self.current_model_dir / "model.cp")
        self.algo.save_to_path(str((self.current_model_dir / "algo_checkpoint").absolute()))
        
        print(f"✓ Model saved to: {self.current_model_dir}")
        return self.current_model_dir
    
    def get_latest_checkpoint(self, model_dir: Path) -> Optional[Path]:
        checkpoints_dir = model_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        
        checkpoints = list(checkpoints_dir.glob("model_iter_*.cp"))
        if not checkpoints:
            return None
        
        def get_iter(p: Path) -> int:
            return int(p.stem.split("_")[-1])
        
        return max(checkpoints, key=get_iter)
    
    def load_model(self, model_name: str) -> None:
        from ray.rllib.algorithms.ppo import PPO
        
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{model_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        self.current_model_dir = model_dir
        self.algo = PPO.from_checkpoint(str((model_dir / "algo_checkpoint").absolute()))
        print(f"✓ Algorithm loaded from: {model_dir}")
    
    def load_backbone_weights(self, model_name: str) -> Dict[str, Any]:
        model_dir = MODELS_DIR / f"rllib_{self.config.algorithm.name}_{model_name}"
        checkpoint_path = model_dir / "model.cp"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
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
        
        iterations = self.metrics["iteration"]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, self.metrics["episode_len_mean"], "g-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Episode Length Mean")
        ax.set_title("Episode Length")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "episode_length.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        win_rates = self.metrics["win_rate_vs_random"]
        eval_iters = [i for i, w in zip(iterations, win_rates) if w is not None]
        eval_wins = [w for w in win_rates if w is not None]
        if eval_iters:
            ax.plot(eval_iters, eval_wins, "b-o", linewidth=1.5, markersize=4, label="Win rate")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate vs Random Agent")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "win_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, self.metrics["total_loss"], "r-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Loss")
        ax.set_title("PPO Total Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "total_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, self.metrics["policy_loss"], "m-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Policy Loss")
        ax.set_title("PPO Policy Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "policy_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, self.metrics["vf_loss"], "c-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value Function Loss")
        ax.set_title("PPO Value Function Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "vf_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iterations, self.metrics["policy_entropy"], "y-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "policy_entropy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        kl_data = [(i, v) for i, v in zip(iterations, self.metrics["kl_divergence"]) if v is not None]
        if kl_data:
            kl_iters, kl_values = zip(*kl_data)
            ax.plot(kl_iters, kl_values, "k-", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL Divergence")
        ax.set_title("Policy KL Divergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "kl_divergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        vf_var = self.metrics["vf_explained_var"]
        ax.plot(iterations, vf_var, "orange", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Value Function Explained Variance")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "vf_explained_variance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        episode_lens = self.metrics["episode_len_mean"]
        num_bins = min(len(episode_lens), 15) if episode_lens else 1
        if episode_lens:
            ax.hist(episode_lens, bins=num_bins, color="purple", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Episode Length")
        ax.set_ylabel("Frequency")
        ax.set_title("Recent Episode Length Distribution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "episode_length_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"\n📊 Graphs saved to: {graphs_dir}\n")
