"""Train SCS_Game agents using Ray RLlib.

This script wraps the ResNet architecture from neural_networks/architectures
for use with RLlib's TorchModelV2 interface.

The environment uses:
- action_mask_location="info" (standard approach)
- obs_space_format="channels_first" (PyTorch convention, required for the ResNet)
"""
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.registry import register_env
from rl_scs import SCS_Game

from neural_networks.architectures.ResNet import ResNet

torch, nn = try_import_torch()


class SCSModel(TorchModelV2, nn.Module):
    """
    RLlib-compatible wrapper around the ResNet architecture for SCS_Game.
    
    This model:
    1. Uses ResNet with conv layers for spatial feature extraction
    2. Outputs policy logits and value estimate
    3. Applies action masking from the observation Dict
    
    Expects observation space to be a Dict with:
    - "observation": Box of shape (C, H, W)
    - "action_mask": MultiBinary(num_actions)
    """
    
    def __init__(
        self,
        obs_space,
        action_space: Discrete,
        num_outputs: int,
        model_config: Dict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        
        # Get custom model config
        custom_config = model_config.get("custom_model_config", {})
        
        # Handle Dict observation space: extract the actual observation shape
        # obs_space could be Dict or original_space could be set
        original_space = getattr(obs_space, "original_space", obs_space)
        if hasattr(original_space, 'spaces') and "observation" in original_space.spaces:
            # Dict space - get the "observation" component
            actual_obs_space = original_space.spaces["observation"]
            self.obs_shape = actual_obs_space.shape
        elif hasattr(obs_space, 'spaces') and "observation" in obs_space.spaces:
            actual_obs_space = obs_space.spaces["observation"]
            self.obs_shape = actual_obs_space.shape
        else:
            self.obs_shape = obs_space.shape
        
        in_channels = self.obs_shape[0]
        rows = self.obs_shape[1]
        cols = self.obs_shape[2]
        
        # Policy output shape: (action_planes, rows, cols) flattened = num_outputs
        # We need to figure out action_planes from num_outputs
        policy_channels = num_outputs // (rows * cols)
        
        # Build ResNet backbone
        self.backbone = ResNet(
            in_channels=in_channels,
            policy_channels=policy_channels,
            num_filters=custom_config.get("num_filters", 128),
            num_blocks=custom_config.get("num_blocks", 4),
            batch_norm=custom_config.get("batch_norm", False),
            hex=custom_config.get("hex", False),  # Use regular conv for now
        )
        
        self._value = None
    
    def forward(
        self,
        input_dict: Dict,
        state: List,
        seq_lens,
    ):
        # Handle Dict observation: extract "observation" and "action_mask"
        obs_input = input_dict["obs"]
        
        if isinstance(obs_input, dict):
            obs = obs_input["observation"].float()
            action_mask = obs_input["action_mask"]
        else:
            obs = obs_input.float()
            action_mask = None
        
        # Forward through ResNet backbone
        policy_logits, value = self.backbone(obs)
        
        # Flatten policy output: (batch, channels, rows, cols) -> (batch, num_actions)
        policy_logits = policy_logits.reshape(policy_logits.shape[0], -1)
        
        # Store value for value_function()
        self._value = value.squeeze(-1)
        
        # Apply action mask to set invalid actions to -inf
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask.float() + 1e-10), min=FLOAT_MIN)
            policy_logits = policy_logits + inf_mask
        
        return policy_logits, state
    
    def value_function(self):
        assert self._value is not None, "Must call forward() first"
        return self._value


# Configuration
CONFIG_PATH = "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/src/example_configurations/randomized_config_5.yml"
ENV_NAME = "scs_game"


def plot_training_progress(metrics: Dict, algorithm: str = "PPO") -> None:
    """Plot training metrics and save to file."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"SCS_Game {algorithm} Training Progress", fontsize=14, fontweight="bold")
    
    iterations = metrics["iteration"]
    
    # Episode Reward Mean
    ax1 = axes[0, 0]
    ax1.plot(iterations, metrics["episode_reward_mean"], "b-", linewidth=1.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Episode Reward Mean")
    ax1.set_title("Reward over Training")
    ax1.grid(True, alpha=0.3)
    
    # Episode Length Mean
    ax2 = axes[0, 1]
    ax2.plot(iterations, metrics["episode_len_mean"], "g-", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Episode Length Mean")
    ax2.set_title("Episode Length over Training")
    ax2.grid(True, alpha=0.3)
    
    # Timesteps Total
    ax3 = axes[1, 0]
    ax3.plot(iterations, metrics["timesteps_total"], "r-", linewidth=1.5)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Total Timesteps")
    ax3.set_title("Cumulative Timesteps")
    ax3.grid(True, alpha=0.3)
    
    # Reward vs Length scatter
    ax4 = axes[1, 1]
    ax4.scatter(metrics["episode_len_mean"], metrics["episode_reward_mean"], 
                c=iterations, cmap="viridis", alpha=0.7)
    ax4.set_xlabel("Episode Length Mean")
    ax4.set_ylabel("Episode Reward Mean")
    ax4.set_title("Reward vs Episode Length")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"training_progress_{algorithm.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Training plot saved to: {output_path}")
    
    plt.show()


def env_creator(config: Dict, debug: bool = False):
    """Create SCS_Game environment wrapped for RLlib."""
    game_config = config.get("game_config", CONFIG_PATH)
    # Use channels_first for PyTorch/ResNet compatibility
    # Use action_mask_location="obs" so mask is in observation Dict
    env = SCS_Game(
        game_config,
        action_mask_location="obs",  # Action mask in observation for RLlib
        obs_space_format="channels_first",
        debug=debug,
    )
    env.simulation_mode = False
    return PettingZooEnv(env)


def env_creator_debug(config: Dict):
    """Create SCS_Game environment with debug=True."""
    return env_creator(config, debug=True)


def train_ppo(num_iterations: int = 10, debug: bool = True):
    """
    Train SCS_Game with PPO algorithm.
    
    Args:
        num_iterations: Number of training iterations
        debug: Enable debug output from environment
    """
    print("=" * 70)
    print("Training SCS_Game with PPO")
    print(f"  Iterations: {num_iterations}, Debug: {debug}")
    print("=" * 70)
    
    # Register environment and custom model
    env_fn = env_creator_debug if debug else env_creator
    register_env(ENV_NAME, env_fn)
    ModelCatalog.register_custom_model("scs_model", SCSModel)
    
    # Create test environment to get spaces
    test_env = env_fn({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    print(f"\nEnvironment Info:")
    print(f"  Observation space: {obs_space['player_0']}")
    print(f"  Action space: {act_space['player_0']}")
    
    single_agent_obs_space = obs_space["player_0"]
    single_agent_act_space = act_space["player_0"]
    
    # Configure PPO with old API stack (required for PettingZoo AEC)
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=ENV_NAME,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=0,  # Run locally for debugging
            rollout_fragment_length=64,
        )
        .training(
            train_batch_size=128,
            lr=3e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            model={
                "custom_model": "scs_model",
                "custom_model_config": {
                    "num_filters": 128,
                    "num_blocks": 2,
                    "batch_norm": False,
                    "hex": False,
                },
            },
        )
        .framework("torch")
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    single_agent_obs_space,
                    single_agent_act_space,
                    {},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=0)
        .debugging(log_level="DEBUG" if debug else "WARN")
    )
    
    print("\nBuilding PPO algorithm...")
    algo = config.build()
    print("✓ Algorithm built successfully!")
    
    # Training metrics storage
    metrics = {
        "iteration": [],
        "episode_reward_mean": [],
        "episode_len_mean": [],
        "timesteps_total": [],
    }
    
    # Training loop
    print(f"\nStarting training for {num_iterations} iterations...")
    print("-" * 70)
    
    for i in range(num_iterations):
        result = algo.train()
        
        env_runners = result.get("env_runners", result)
        episode_reward_mean = env_runners.get("episode_reward_mean", float("nan"))
        episode_len_mean = env_runners.get("episode_len_mean", float("nan"))
        timesteps_total = result.get("timesteps_total", 0)
        
        # Store metrics
        metrics["iteration"].append(i + 1)
        metrics["episode_reward_mean"].append(episode_reward_mean if not np.isnan(episode_reward_mean) else 0)
        metrics["episode_len_mean"].append(episode_len_mean if not np.isnan(episode_len_mean) else 0)
        metrics["timesteps_total"].append(timesteps_total)
        
        print(
            f"Iter {i+1:3d}/{num_iterations} | "
            f"Reward: {episode_reward_mean:7.3f} | "
            f"EpLen: {episode_len_mean:6.1f} | "
            f"Steps: {timesteps_total:6d}"
        )
    
    print("-" * 70)
    print("✓ PPO Training completed!")
    algo.stop()
    
    # Plot training progress
    plot_training_progress(metrics, "PPO")
    return algo


def train_impala(num_iterations: int = 10, debug: bool = True):
    """
    Train SCS_Game with IMPALA algorithm.
    
    Args:
        num_iterations: Number of training iterations
        debug: Enable debug output from environment
    """
    print("=" * 70)
    print("Training SCS_Game with IMPALA")
    print(f"  Iterations: {num_iterations}, Debug: {debug}")
    print("=" * 70)
    
    # Register environment and custom model
    env_fn = env_creator_debug if debug else env_creator
    register_env(ENV_NAME, env_fn)
    ModelCatalog.register_custom_model("scs_model", SCSModel)
    
    # Create test environment to get spaces
    test_env = env_fn({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    print(f"\nEnvironment Info:")
    print(f"  Observation space: {obs_space['player_0']}")
    print(f"  Action space: {act_space['player_0']}")
    
    single_agent_obs_space = obs_space["player_0"]
    single_agent_act_space = act_space["player_0"]
    
    # Configure IMPALA with old API stack (required for PettingZoo AEC)
    config = (
        ImpalaConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=ENV_NAME,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=32,
        )
        .training(
            train_batch_size=64,
            lr=1e-4,
            gamma=0.99,
            model={
                "custom_model": "scs_model",
                "custom_model_config": {
                    "num_filters": 32,
                    "num_blocks": 1,
                    "batch_norm": False,
                    "hex": True,
                },
            },
        )
        .framework("torch")
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    single_agent_obs_space,
                    single_agent_act_space,
                    {},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .resources(num_gpus=0)
        .debugging(log_level="DEBUG" if debug else "WARN")
    )
    
    print("\nBuilding IMPALA algorithm...")
    algo = config.build()
    print("✓ Algorithm built successfully!")
    
    # Training metrics storage
    metrics = {
        "iteration": [],
        "episode_reward_mean": [],
        "episode_len_mean": [],
        "timesteps_total": [],
    }
    
    # Training loop
    print(f"\nStarting training for {num_iterations} iterations...")
    print("-" * 70)
    
    for i in range(num_iterations):
        result = algo.train()
        
        env_runners = result.get("env_runners", result)
        episode_reward_mean = env_runners.get("episode_reward_mean", float("nan"))
        episode_len_mean = env_runners.get("episode_len_mean", float("nan"))
        timesteps_total = result.get("timesteps_total", 0)
        
        # Store metrics
        metrics["iteration"].append(i + 1)
        metrics["episode_reward_mean"].append(episode_reward_mean if not np.isnan(episode_reward_mean) else 0)
        metrics["episode_len_mean"].append(episode_len_mean if not np.isnan(episode_len_mean) else 0)
        metrics["timesteps_total"].append(timesteps_total)
        
        print(
            f"Iter {i+1:3d}/{num_iterations} | "
            f"Reward: {episode_reward_mean:7.3f} | "
            f"EpLen: {episode_len_mean:6.1f} | "
            f"Steps: {timesteps_total:6d}"
        )
    
    print("-" * 70)
    print("✓ IMPALA Training completed!")
    algo.stop()
    
    # Plot training progress
    plot_training_progress(metrics, "IMPALA")
    return algo


def main():
    """Main entry point - choose which algorithm to test."""
    # Initialize Ray once
    ray.init(ignore_reinit_error=True, num_cpus=4)
    
    try:
        # Choose algorithm to test:
        train_ppo(num_iterations=100, debug=False)
        # train_impala(num_iterations=5, debug=False)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ray.shutdown()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
