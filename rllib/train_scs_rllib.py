import random
from pathlib import Path
from typing import Dict, List, Optional

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
MODELS_DIR = Path(__file__).parent.parent / "models"


class SCSTrainer:
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        self.algo = None
        self.metrics: Dict[str, List] = {}
        
        # Register environment
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
    
    
    def train_ppo(self, num_iterations: int = 10, model_name: Optional[str] = None) -> None:
        print("=" * 70)
        print("Training SCS_Game with PPO (new API stack)")
        print(f"  Iterations: {num_iterations}, Debug: {self.debug}")
        print("=" * 70)
        
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
                minibatch_size=64,
                lr=1e-4,
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
                                    "num_blocks": 2,
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
        
        print("\nBuilding PPO algorithm...")
        self.algo = config.build_algo()
        print("✓ Algorithm built successfully!")
        
        self.metrics = {
            "iteration": [],
            "episode_len_mean": [],
            "num_env_steps_sampled_lifetime": [],
            "win_rate_vs_random": [],
        }
        
        print(f"\nStarting training for {num_iterations} iterations...")
        print("-" * 70)
        
        for i in range(num_iterations):
            result = self.algo.train()
            
            env_runners = result.get("env_runners", {})
            episode_len_mean = env_runners.get("episode_len_mean", 0) or 0
            num_steps = int(result.get("num_env_steps_sampled_lifetime", 0))
            
            win_rate = None
            if (i + 1) % 10 == 0:
                win_rate = self.evaluate_vs_random(num_games=50)
            
            self.metrics["iteration"].append(i + 1)
            self.metrics["episode_len_mean"].append(episode_len_mean)
            self.metrics["num_env_steps_sampled_lifetime"].append(num_steps)
            self.metrics["win_rate_vs_random"].append(win_rate)
            
            win_str = f" | WinRate: {win_rate:.1%}" if win_rate is not None else ""
            print(f"Iter {i+1:3d}/{num_iterations} | EpLen: {episode_len_mean:6.1f} | Steps: {num_steps:7d}{win_str}")
        
        print("-" * 70)
        print("✓ PPO Training completed!")
        
        if model_name:
            self.save_model(model_name)
        
        self.algo.stop()
        self.plot_progress()
    
    def save_model(self, model_name: str) -> Path:
        if self.algo is None:
            raise RuntimeError("No algorithm to save. Train first.")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODELS_DIR / f"rllib_ppo_{model_name}"
        
        self.algo.save_to_path(str(save_path))
        print(f"✓ Model saved to: {save_path}")
        return save_path
    
    def load_model(self, model_name: str) -> None:
        from ray.rllib.algorithms.ppo import PPO
        
        load_path = MODELS_DIR / f"rllib_ppo_{model_name}"
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        self.algo = PPO.from_checkpoint(str(load_path))
        print(f"✓ Model loaded from: {load_path}")


    
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
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("SCS_Game Training Progress", fontsize=14, fontweight="bold")
        
        iterations = self.metrics["iteration"]
        
        # Episode Length Mean
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.metrics["episode_len_mean"], "g-", linewidth=1.5)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Episode Length Mean")
        ax1.set_title("Episode Length (longer = better play)")
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        win_rates = self.metrics["win_rate_vs_random"]
        eval_iters = [i for i, w in zip(iterations, win_rates) if w is not None]
        eval_wins = [w for w in win_rates if w is not None]
        if eval_iters:
            ax2.plot(eval_iters, eval_wins, "b-o", linewidth=1.5, markersize=4, label="Win rate")
        ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Random baseline")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Win Rate vs Random Agent")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.metrics["num_env_steps_sampled_lifetime"], "r-", linewidth=1.5)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Total Timesteps")
        ax3.set_title("Cumulative Timesteps")
        ax3.grid(True, alpha=0.3)
        
        # Episode Length distribution (last N)
        ax4 = axes[1, 1]
        recent_lens = self.metrics["episode_len_mean"][-20:] if len(self.metrics["episode_len_mean"]) > 20 else self.metrics["episode_len_mean"]
        ax4.hist(recent_lens, bins=10, color="purple", alpha=0.7, edgecolor="black")
        ax4.set_xlabel("Episode Length")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Recent Episode Length Distribution")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = "training_progress.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Training plot saved to: {output_path}")
        plt.close(fig)
