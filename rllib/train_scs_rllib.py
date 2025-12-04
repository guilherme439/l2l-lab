import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from rl_scs import SCS_Game

from neural_networks.architectures.dual_head import ResNet

from .dual_head_torch_model import DualHeadRLModule

DEFAULT_CONFIG_PATH = "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/src/example_configurations/mirrored_config_5.yml"
ENV_NAME = "scs_game"
MODELS_DIR = Path("models")


class SCSTrainer:
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        self.algo = None
        self.metrics: Dict[str, List] = {}
        self.current_model_dir: Optional[Path] = None
        self._register_env()
    
    def _register_env(self) -> None:
        def env_creator(config: Dict):
            game_config = config.get("game_config", self.config_path)
            env = SCS_Game(
                game_config,
                action_mask_location="obs",
                obs_space_format="channels_first",
                debug=self.debug,
            )
            env.simulation_mode = False
            return PettingZooEnv(env)
        
        register_env(ENV_NAME, env_creator)
    
    def _get_spaces(self):
        env = SCS_Game(
            self.config_path,
            action_mask_location="obs",
            obs_space_format="channels_first",
        )
        wrapped = PettingZooEnv(env)
        obs_space = wrapped.observation_space["player_0"]
        act_space = wrapped.action_space["player_0"]
        return obs_space, act_space
    
    def _setup_model_dir(self, model_name: str) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = MODELS_DIR / f"rllib_ppo_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        self.current_model_dir = model_dir
        return model_dir
    
    
    def train_ppo(self, num_iterations: int, model_name: str) -> None:
        print("\n" * 5)
        print("=" * 70)
        print("Training SCS_Game with PPO (new API stack)")
        print(f"  Iterations: {num_iterations}, Debug: {self.debug}")
        print("=" * 70)
        
        self._setup_model_dir(model_name)
        
        obs_space, act_space = self._get_spaces()
        
        print(f"\nEnvironment Info:")
        print(f"  Observation space: {obs_space}")
        print(f"  Action space: {act_space}")
        
        config = (
            PPOConfig()
            .environment(
                env=ENV_NAME,
                disable_env_checking=True,
            )
            .env_runners(
                num_env_runners=0,
                rollout_fragment_length=32,
            )
            .training(
                train_batch_size_per_learner=256,
                minibatch_size=32,
                lr=2e-4,
                gamma=0.99,
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
                                "network_class": ResNet,
                                "network_kwargs": {
                                    "num_filters": 64,
                                    "num_blocks": 4,
                                    "batch_norm": False,
                                    "hex": False,
                                },
                            },
                        ),
                    },
                ),
            )
            .framework("torch")
            .resources(num_gpus=0)
            .debugging(log_level="DEBUG" if self.debug else "WARN")
        )
        
        print("\nBuilding PPO algorithm...\n\n")
        self.algo = config.build_algo()
        print("\n\n✓ Algorithm built successfully!")
        
        self.metrics = {
            "iteration": [],
            "episode_len_mean": [],
            "total_loss": [],
            "win_rate_vs_random": [],
        }
        
        print(f"\nStarting training for {num_iterations} iterations...")
        print("-" * 70)
        
        for i in range(num_iterations):
            result = self.algo.train()
            
            env_runners = result.get("env_runners", {})
            episode_len_mean = env_runners.get("episode_len_mean", 0) or 0
            total_loss = result.get("learners", {}).get("shared_policy", {}).get("total_loss", 0) or 0

            win_rate = None
            if (i + 1) % 20 == 0:
                win_rate = self.evaluate_vs_random(num_games=50)
            
            self.metrics["iteration"].append(i + 1)
            self.metrics["episode_len_mean"].append(episode_len_mean)
            self.metrics["total_loss"].append(total_loss)
            self.metrics["win_rate_vs_random"].append(win_rate)
            
            wr_str = f" => WinRate: {win_rate:.1%}" if win_rate is not None else ""
            print(f"{i+1:3d}/{num_iterations} | EpLen: {episode_len_mean:6.1f} {wr_str}")
        
        print("-" * 70)
        print("✓ PPO Training completed!")
        
        self.save_model(model_name)
        
        self.algo.stop()
        self.plot_progress()
    
    def save_model(self, model_name: str) -> Path:
        if self.algo is None:
            raise RuntimeError("No algorithm to save. Train first.")
        
        model_dir = MODELS_DIR / f"rllib_ppo_{model_name}"
        if not model_dir.exists():
            raise RuntimeError(f"Model directory not found: {model_dir}. Call _setup_model_dir first.")
        self.current_model_dir = model_dir
        
        rl_module = self.algo.get_module("shared_policy")
        checkpoint = {
            "backbone_state_dict": rl_module.backbone.state_dict(),
            "observation_space": rl_module.observation_space,
            "action_space": rl_module.action_space,
            "model_config": rl_module.model_config,
        }
        
        torch.save(checkpoint, model_dir / "model.cp")
        self.algo.save_to_path(str((model_dir / "algo_checkpoint").absolute()))
        
        print(f"✓ Model saved to: {model_dir}")
        return model_dir
    
    def load_model(self, model_name: str) -> None:
        from ray.rllib.algorithms.ppo import PPO
        
        model_dir = MODELS_DIR / f"rllib_ppo_{model_name}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        self.current_model_dir = model_dir
        self.algo = PPO.from_checkpoint(str((model_dir / "algo_checkpoint").absolute()))
        print(f"✓ Algorithm loaded from: {model_dir}")
    
    def load_backbone_weights(self, model_name: str) -> Dict[str, Any]:
        model_dir = MODELS_DIR / f"rllib_ppo_{model_name}"
        checkpoint_path = model_dir / "model.cp"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"✓ Backbone weights loaded from: {checkpoint_path}")
        return checkpoint


    
    def evaluate_vs_random(self, num_games: int = 10) -> float:
        if self.algo is None:
            print("No algorithm trained yet!")
            return 0.0
        
        rl_module = self.algo.get_module("shared_policy")
        rl_module.eval()
        
        wins = 0
        env = SCS_Game(
            self.config_path,
            action_mask_location="obs",
            obs_space_format="channels_first",
        )
        
        for _ in range(num_games):
            env.reset()
            
            while not env.terminal:
                agent = env.agent_selection
                obs = env.observe(agent)
                action_mask = obs["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]
                
                if len(valid_actions) == 0:
                    env.step(0)
                    continue
                
                if agent == "player_0":
                    obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
                    mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        batch = {"obs": {"observation": obs_tensor, "action_mask": mask_tensor}}
                        output = rl_module.forward_inference(batch)
                        logits = output["action_dist_inputs"].squeeze(0)
                        action = int(torch.argmax(logits).item())
                else:
                    action = random.choice(valid_actions)
                
                env.step(action)
            
            if env.terminal_value > 0:
                wins += 1
        
        rl_module.train()
        return wins / num_games
    
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
        ax.set_title("Episode Length (longer = better play)")
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
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random baseline")
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
        recent_lens = self.metrics["episode_len_mean"][-20:] if len(self.metrics["episode_len_mean"]) > 20 else self.metrics["episode_len_mean"]
        num_bins = min(len(recent_lens), 10)
        ax.hist(recent_lens, bins=num_bins, color="purple", alpha=0.7, edgecolor="black", rwidth=0.8)
        ax.set_xlabel("Episode Length")
        ax.set_ylabel("Frequency")
        ax.set_title("Recent Episode Length Distribution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "episode_length_dist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"\n📊 Graphs saved to: {graphs_dir}")
