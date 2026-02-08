from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

from configs.definition.common.EnvConfig import EnvConfig
from configs.definition.testing.agents.AgentConfig import AgentConfig
from configs.definition.testing.agents.PolicyAgentConfig import \
    PolicyAgentConfig
from configs.definition.testing.agents.RandomAgentConfig import \
    RandomAgentConfig
from configs.definition.testing.TestingConfig import TestingConfig
from configs.definition.training.NetworkConfig import (CONV_ARCHITECTURES,
                                                       MLP_ARCHITECTURES)
from envs.registry import create_env

MODELS_DIR = Path("models")



@dataclass
class GameResults:
    wins: int
    losses: int
    draws: int
    total: int
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0
    
    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total > 0 else 0.0
    
    @property
    def draw_rate(self) -> float:
        return self.draws / self.total if self.total > 0 else 0.0


class Tester:
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = TestingConfig.from_yaml(config_path)
    
    @staticmethod
    def _get_checkpoint_path(model_name: str, checkpoint: int) -> Path:
        model_dir = MODELS_DIR / f"{model_name}"
        return model_dir / "checkpoints" / str(checkpoint) / "model.cp"

    @staticmethod
    def _create_conv_backbone(
        network_class,
        network_kwargs: Dict[str, Any],
        inner_obs_space,
        act_space,
    ) -> torch.nn.Module:
        obs_shape = inner_obs_space.shape
        in_channels = obs_shape[0]
        rows, cols = obs_shape[1], obs_shape[2]
        policy_channels = act_space.n // (rows * cols)
        return network_class(
            in_channels=in_channels,
            policy_channels=policy_channels,
            **network_kwargs,
        )

    @staticmethod
    def _create_mlp_backbone(
        network_class,
        network_kwargs: Dict[str, Any],
        inner_obs_space,
        act_space,
    ) -> torch.nn.Module:
        out_features = act_space.n
        backbone = network_class(
            out_features=out_features,
            **network_kwargs,
        )
        dummy_obs = torch.zeros((1,) + inner_obs_space.shape, dtype=torch.float32)
        with torch.no_grad():
            _ = backbone(dummy_obs)
        return backbone

    def _create_backbone(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        model_config = checkpoint["model_config"]
        network_class = model_config["network_class"]
        network_kwargs = model_config.get("network_kwargs", {})
        architecture = model_config.get("architecture")
        
        obs_space = checkpoint["observation_space"]
        act_space = checkpoint["action_space"]
        
        inner_obs_space = obs_space["observation"]
        
        if architecture in CONV_ARCHITECTURES:
            backbone = Tester._create_conv_backbone(network_class, network_kwargs, inner_obs_space, act_space)
        elif architecture in MLP_ARCHITECTURES:
            backbone = Tester._create_mlp_backbone(network_class, network_kwargs, inner_obs_space, act_space)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
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
    
    @staticmethod
    def _is_env_done(env) -> bool:
        return all(env.terminations.values()) or all(env.truncations.values())
    
    @staticmethod
    def _get_game_result(env) -> int:
        rewards = env.rewards
        agents = list(rewards.keys())
        if len(agents) < 2:
            return 0
        r0, r1 = rewards.get(agents[0], 0), rewards.get(agents[1], 0)
        if r0 > r1:
            return 1
        elif r0 < r1:
            return -1
        return 0
    
    def play_game(self, backbone_1: torch.nn.Module, backbone_2: torch.nn.Module):
        env_config = self.config.env
        env = create_env(env_config.name, **env_config.kwargs)
        env.reset()
        
        backbones = {"player_0": backbone_1, "player_1": backbone_2}
        
        while not self._is_env_done(env):
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
    
    def _create_action_fn(self, agent_config: AgentConfig):
        if isinstance(agent_config, RandomAgentConfig):
            return lambda obs: random.choice(np.where(obs["action_mask"] == 1)[0])
        
        if isinstance(agent_config, PolicyAgentConfig):
            cp_path = self._get_checkpoint_path(agent_config.model_name, agent_config.checkpoint)
            backbone, obs_format = Tester._load_backbone_from_checkpoint(cp_path)
            return lambda obs: Tester._sample_action_from_backbone(backbone, obs, obs_format)
        
        raise ValueError(f"Unknown agent config type: {type(agent_config)}")
    
    def _agent_description(self, agent_config: AgentConfig) -> str:
        if isinstance(agent_config, RandomAgentConfig):
            return "random"
        if isinstance(agent_config, PolicyAgentConfig):
            return f"{agent_config.model_name}@{agent_config.checkpoint}"
        return "unknown"
    
    def test(self) -> GameResults:
        print("\n" + "=" * 70)
        print("Testing Mode")
        print(f"  Player 1: {self._agent_description(self.config.p1)}")
        print(f"  Player 2: {self._agent_description(self.config.p2)}")
        print(f"  Games: {self.config.num_games}")
        print("=" * 70)
        
        get_action_p1 = self._create_action_fn(self.config.p1)
        get_action_p2 = self._create_action_fn(self.config.p2)
        
        results = Tester._run_games(
            self.config.env, self.config.num_games,
            get_action_p0=get_action_p1,
            get_action_p1=get_action_p2,
        )
        
        print(f"\nResults (P1 perspective):")
        print(f"  Wins:   {results.wins} ({results.win_rate:.1%})")
        print(f"  Losses: {results.losses} ({results.loss_rate:.1%})")
        print(f"  Draws:  {results.draws} ({results.draw_rate:.1%})")
        print("\n✓ Testing complete!")
        
        return results

    @staticmethod
    def _sample_action_from_backbone(backbone, obs: Dict[str, np.ndarray], obs_space_format: str = "channels_first") -> int:
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        action_mask = obs["action_mask"]
        
        if obs_space_format == "channels_last":
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            policy_logits, _ = backbone(obs_tensor)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        
        return action
    
    @staticmethod
    def _sample_action_from_rl_module(rl_module, obs: Dict[str, np.ndarray]) -> int:
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        action_mask = obs["action_mask"]
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            batch = {"obs": {"observation": obs_tensor, "action_mask": mask_tensor}}
            output = rl_module.forward_inference(batch)
            logits = output["action_dist_inputs"].squeeze(0)
            logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        
        return action
    
    @staticmethod
    def _load_backbone_from_checkpoint(checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model_config = checkpoint["model_config"]
        architecture = model_config.get("architecture", "ConvNet")
        obs_space_format = model_config.get("obs_space_format", "channels_first")
        
        obs_space = checkpoint["observation_space"]
        act_space = checkpoint["action_space"]
        inner_obs_space = obs_space["observation"]
        
        if architecture in CONV_ARCHITECTURES:
            backbone = Tester._create_conv_backbone(
                model_config["network_class"], model_config.get("network_kwargs", {}),
                inner_obs_space, act_space
            )
        elif architecture in MLP_ARCHITECTURES:
            backbone = Tester._create_mlp_backbone(
                model_config["network_class"], model_config.get("network_kwargs", {}),
                inner_obs_space, act_space
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        backbone.eval()
        return backbone, obs_space_format
    
    @staticmethod
    def _run_games(env_config: EnvConfig, num_games: int, get_action_p0, get_action_p1) -> GameResults:
        wins, losses, draws = 0, 0, 0
        env = create_env(env_config.name, **env_config.kwargs)
        
        for _ in range(num_games):
            env.reset()
            
            while not Tester._is_env_done(env):
                agent = env.agent_selection
                obs = env.observe(agent)
                valid_actions = np.where(obs["action_mask"] == 1)[0]
                
                if len(valid_actions) == 0:
                    env.step(0)
                    continue
                
                if agent == "player_0":
                    action = get_action_p0(obs)
                else:
                    action = get_action_p1(obs)
                
                env.step(action)
            
            result = Tester._get_game_result(env)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        
        return GameResults(wins=wins, losses=losses, draws=draws, total=num_games)
    
    @staticmethod
    def evaluate_checkpoint_vs_checkpoint(
        checkpoint_1_path: Path,
        checkpoint_2_path: Path,
        env_config: EnvConfig,
        num_games: int = 10
    ) -> GameResults:
        if not checkpoint_1_path.exists() or not checkpoint_2_path.exists():
            return GameResults(0, 0, 0, 0)
        
        backbone_1, obs_format_1 = Tester._load_backbone_from_checkpoint(checkpoint_1_path)
        backbone_2, obs_format_2 = Tester._load_backbone_from_checkpoint(checkpoint_2_path)
        
        return Tester._run_games(
            env_config, num_games,
            get_action_p0=lambda obs: Tester._sample_action_from_backbone(backbone_1, obs, obs_format_1),
            get_action_p1=lambda obs: Tester._sample_action_from_backbone(backbone_2, obs, obs_format_2),
        )
    
    @staticmethod
    def _get_main_module(algo):
        try:
            module = algo.get_module("main_policy")
            if module is not None:
                return module
        except KeyError:
            pass
        return algo.get_module("shared_policy")
    
    @staticmethod
    def evaluate_vs_checkpoint(algo, checkpoint_path: Path, env_config: EnvConfig, num_games: int = 10) -> GameResults:
        if algo is None or not checkpoint_path.exists():
            return GameResults(0, 0, 0, 0)
        
        opponent, obs_format = Tester._load_backbone_from_checkpoint(checkpoint_path)
        
        rl_module = Tester._get_main_module(algo)
        rl_module.eval()
        
        results = Tester._run_games(
            env_config, num_games,
            get_action_p0=lambda obs: Tester._sample_action_from_rl_module(rl_module, obs),
            get_action_p1=lambda obs: Tester._sample_action_from_backbone(opponent, obs, obs_format),
        )
        
        rl_module.train()
        return results

    @staticmethod
    def evaluate_vs_random(algo, env_config: EnvConfig, num_games: int = 10) -> GameResults:
        if algo is None:
            print("No algorithm trained yet!")
            return GameResults(0, 0, 0, 0)
        
        rl_module = Tester._get_main_module(algo)
        rl_module.eval()
        
        results = Tester._run_games(
            env_config, num_games,
            get_action_p0=lambda obs: Tester._sample_action_from_rl_module(rl_module, obs),
            get_action_p1=lambda obs: random.choice(np.where(obs["action_mask"] == 1)[0]),
        )
        
        rl_module.train()
        return results


