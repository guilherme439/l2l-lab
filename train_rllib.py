"""Train SCS_Game agents using Ray RLlib with PPO

Based on PettingZoo + RLlib examples:
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py
- https://pettingzoo.farama.org/tutorials/rllib/holdem/
"""
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from SCS_Game import SCS_Game


def env_creator(config):
    """Create SCS_Game environment wrapped for RLlib"""
    game_config = config.get(
        "game_config", 
        "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/Game_configs/randomized_config_5.yml"
    )
    return PettingZooEnv(SCS_Game(game_config))


def main():
    print("=" * 60)
    print("Training SCS_Game with Ray RLlib PPO")
    print("=" * 60)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Register the environment
        env_name = "scs_game"
        register_env(env_name, lambda config: env_creator(config))
        
        # Create a test environment to inspect spaces
        test_env = env_creator({})
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        
        print(f"\nEnvironment Configuration:")
        print(f"  Observation space: {obs_space}")
        print(f"  Action space: {act_space}")
        print(f"  Number of agents: {len(test_env.get_agent_ids())}")
        
        # Configure PPO for multi-agent training  
        # Using old API stack as it's more compatible with PettingZoo
        config = {
            "env": env_name,
            "env_config": {},
            "framework": "torch",
            "num_workers": 0,  # Run in main process for simplicity
            "lr": 5e-5,
            "gamma": 0.99,
            "model": {
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            },
            "multiagent": {
                "policies": {
                    "player_0": (None, obs_space, act_space, {}),
                    "player_1": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
            },
            "_disable_preprocessor_api": True,  # Use raw observations
        }
        
        print("\nBuilding PPO algorithm...")
        from ray.rllib.algorithms.ppo import PPO
        algo = PPO(config=config)
        
        print("\nStarting training...")
        num_iterations = 5
        
        for i in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Training Iteration {i+1}/{num_iterations}")
            print(f"{'='*60}")
            
            result = algo.train()
            
            # Extract metrics
            episode_reward_mean = result.get("episode_reward_mean", "N/A")
            episodes_this_iter = result.get("episodes_this_iter", "N/A")
            timesteps_total = result.get("timesteps_total", "N/A")
            time_this_iter = result.get("time_this_iter_s", 0)
            
            print(f"  Episode reward mean: {episode_reward_mean}")
            print(f"  Episodes this iter: {episodes_this_iter}")
            print(f"  Timesteps total: {timesteps_total}")
            print(f"  Training time: {time_this_iter:.2f}s")
        
        print("\n" + "="*60)
        print("✓ Training completed successfully!")
        print("="*60)
        
        # Test the trained policies
        print("\nTesting trained policies...")
        test_env_for_inference = SCS_Game(
            "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/Game_configs/randomized_config_5.yml"
        )
        test_env_for_inference.reset()
        
        episode_rewards = {agent: 0 for agent in test_env_for_inference.agents}
        step_count = 0
        max_steps = 200
        
        for agent in test_env_for_inference.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = test_env_for_inference.last()
            
            episode_rewards[agent] += reward
            
            if termination or truncation:
                action = None
            else:
                # Compute action using trained policy
                action = algo.compute_single_action(
                    observation, 
                    policy_id=agent,
                    explore=False  # Use deterministic policy for testing
                )
            
            test_env_for_inference.step(action)
            step_count += 1
        
        print(f"\nTest episode finished after {step_count} steps")
        print(f"Episode rewards: {episode_rewards}")
        print(f"Winner: Player {test_env_for_inference.get_winner()}")
        print("✓ Policy test completed!")
        
        # Cleanup
        algo.stop()
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ray.shutdown()
    
    print("\n" + "=" * 60)
    print("RLlib Integration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
