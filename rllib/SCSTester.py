from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from rl_scs.SCS_Game import SCS_Game
from rl_scs.render.SCS_Renderer import SCS_Renderer

from configs.definition.TestingConfig import TestingConfig

MODELS_DIR = Path("models")


class SCSTester:
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = TestingConfig.from_yaml(config_path)
        self.model_dir = MODELS_DIR / f"rllib_ppo_{self.config.model_name}"
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
    
    def _load_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        if checkpoint_name == "latest":
            checkpoint_path = self.model_dir / "model.cp"
        else:
            checkpoint_path = self.model_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path, weights_only=False)
    
    def _create_backbone(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        model_config = checkpoint["model_config"]
        network_class = model_config["network_class"]
        network_kwargs = model_config.get("network_kwargs", {})
        
        obs_space = checkpoint["observation_space"]
        act_space = checkpoint["action_space"]
        
        inner_obs_space = obs_space["observation"]
        obs_shape = inner_obs_space.shape
        in_channels = obs_shape[0]
        rows, cols = obs_shape[1], obs_shape[2]
        policy_channels = act_space.n // (rows * cols)
        
        backbone = network_class(
            in_channels=in_channels,
            policy_channels=policy_channels,
            **network_kwargs,
        )
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        backbone.eval()
        return backbone
    
    def _get_action(self, backbone: torch.nn.Module, obs: Dict[str, np.ndarray]) -> int:
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        action_mask = obs["action_mask"]
        
        with torch.no_grad():
            policy_logits, _ = backbone(obs_tensor)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            
            masked_logits = policy_logits.numpy()
            masked_logits[action_mask == 0] = -np.inf
            action = int(np.argmax(masked_logits))
        
        return action
    
    def play_game(self, backbone_1: torch.nn.Module, backbone_2: torch.nn.Module) -> SCS_Game:
        env = SCS_Game(
            self.config.game_config,
            action_mask_location="obs",
            obs_space_format="channels_first",
        )
        env.reset()
        
        backbones = {"player_0": backbone_1, "player_1": backbone_2}
        
        while not env.terminal:
            agent = env.agent_selection
            obs = env.observe(agent)
            action_mask = obs["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]
            
            if len(valid_actions) == 0:
                env.step(None)
                continue
            
            backbone = backbones[agent]
            action = self._get_action(backbone, obs)
            env.step(action)
        
        return env
    
    def test(self) -> None:
        print("\n" + "=" * 70)
        print("Testing Mode - Checkpoint vs Checkpoint")
        print(f"  Model: {self.config.model_name}")
        print(f"  Player 1: {self.config.checkpoint_1}")
        print(f"  Player 2: {self.config.checkpoint_2}")
        print(f"  Games: {self.config.num_games}")
        print("=" * 70)
        
        cp1 = self._load_checkpoint(self.config.checkpoint_1)
        cp2 = self._load_checkpoint(self.config.checkpoint_2)
        
        print(f"\n✓ Loaded checkpoint 1 (iter {cp1.get('iteration', '?')})")
        print(f"✓ Loaded checkpoint 2 (iter {cp2.get('iteration', '?')})")
        
        backbone_1 = self._create_backbone(cp1)
        backbone_2 = self._create_backbone(cp2)
        
        renderer = SCS_Renderer()
        
        for game_num in range(self.config.num_games):
            print(f"\n--- Game {game_num + 1}/{self.config.num_games} ---")
            
            finished_game = self.play_game(backbone_1, backbone_2)
            
            winner = finished_game.get_winner()
            if winner == 0:
                result = "Draw"
            elif winner == 1:
                result = f"Player 1 ({self.config.checkpoint_1}) wins"
            else:
                result = f"Player 2 ({self.config.checkpoint_2}) wins"
            
            print(f"Result: {result}")
            print(f"Game length: {finished_game.get_length()} actions")
            
            print("\nOpening game analyzer (use arrow keys to navigate)...")
            renderer.analyse(finished_game)
        
        renderer.close()
        print("\n✓ Testing complete!")

    @staticmethod
    def evaluate_vs_random(algo, game_config: str, num_games: int = 10) -> float:
        if algo is None:
            print("No algorithm trained yet!")
            return 0.0

        rl_module = algo.get_module("shared_policy")
        rl_module.eval()

        wins = 0
        env = SCS_Game(
            game_config,
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
