from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from l2l_lab.agents import Agent, PolicyAgent, RandomAgent
from l2l_lab.configs.common.EnvConfig import EnvConfig
from l2l_lab.configs.testing.agents.AgentConfig import AgentConfig
from l2l_lab.configs.testing.agents.MCTSAgentConfig import MCTSAgentConfig
from l2l_lab.configs.testing.agents.PolicyAgentConfig import \
    PolicyAgentConfig
from l2l_lab.configs.testing.agents.RandomAgentConfig import \
    RandomAgentConfig
from l2l_lab.configs.testing.TestingConfig import TestingConfig
from l2l_lab.configs.training.NetworkConfig import (CONV_ARCHITECTURES,
                                                       MLP_ARCHITECTURES,
                                                       NetworkConfig)
from l2l_lab.envs.registry import create_env

MODELS_DIR = Path("models")



@dataclass
class GameResults:
    wins: int
    losses: int
    draws: int
    total: int
    avg_moves: float = 0.0
    elapsed_time: float = 0.0
    as_p0: Optional['GameResults'] = None
    as_p1: Optional['GameResults'] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total if self.total > 0 else 0.0

    @property
    def avg_time_per_game(self) -> float:
        return self.elapsed_time / self.total if self.total > 0 else 0.0


class Tester:

    def __init__(self, config_path: Union[str, Path]):
        self.config = TestingConfig.from_yaml(config_path)

    @staticmethod
    def play_games(
        p0: Agent,
        p1: Agent,
        env_config: EnvConfig,
        num_games: int,
    ) -> GameResults:
        """
            Play `num_games` games with a fixed (p0, p1) pair.
            Returns results from p0's perspective.
        """
        start_time = time.time()
        wins, losses, draws = 0, 0, 0
        total_moves = 0
        env = create_env(env_config.name, **env_config.kwargs)

        for _ in range(num_games):
            env.reset()
            moves = 0

            while not Tester._is_env_done(env):
                agent_id = env.agent_selection
                action_mask = env.observe(agent_id)["action_mask"]

                if np.sum(action_mask) == 0:
                    print("\nno valid actions found")
                    env.step(None)
                    continue

                current = p0 if agent_id == "player_0" else p1
                action = current.choose_action(env)

                env.step(action)
                if action is not None:
                    moves += 1

            result = Tester._get_game_result(env)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

            total_moves += moves

        elapsed = time.time() - start_time
        avg_moves = (total_moves / num_games) if num_games > 0 else 0.0
        return GameResults(wins, losses, draws, num_games, avg_moves, elapsed)

    def test(self) -> GameResults:
        print("\n" + "=" * 70)
        print("Testing Mode")
        print(f"  Player 1: {self.config.p1.agent_type}")
        print(f"  Player 2: {self.config.p2.agent_type}")
        print(f"  Games: {self.config.num_games}")
        print("=" * 70)

        agent_p1 = self._create_agent(self.config.p1)
        agent_p2 = self._create_agent(self.config.p2)

        print(f"  P1 agent: {agent_p1.name}")
        print(f"  P2 agent: {agent_p2.name}")

        results = Tester.play_games(
            p0=agent_p1,
            p1=agent_p2,
            env_config=self.config.env,
            num_games=self.config.num_games,
        )

        print(f"\nResults (P1 perspective):")
        print(f"  Wins:   {results.wins} ({results.win_rate:.1%})")
        print(f"  Losses: {results.losses} ({results.loss_rate:.1%})")
        print(f"  Draws:  {results.draws} ({results.draw_rate:.1%})")
        print(f"  Time:   {results.elapsed_time:.2f}s ({results.avg_time_per_game:.3f}s/game)")
        print("\n✓ Testing complete!")

        return results

    @staticmethod
    def _get_checkpoint_path(model_name: str, checkpoint: int) -> Path:
        model_dir = MODELS_DIR / f"{model_name}"
        return model_dir / "checkpoints" / str(checkpoint) / "model" / "checkpoint.pt"

    @staticmethod
    def _is_env_done(env) -> bool:
        return all(env.terminations.values()) or all(env.truncations.values())

    @staticmethod
    def _get_game_result(env) -> int:
        rewards = env.rewards
        if "player_0" in rewards and "player_1" in rewards:
            r0, r1 = rewards.get("player_0", 0), rewards.get("player_1", 0)
        else:
            agents = list(rewards.keys())
            if len(agents) < 2:
                return 0
            r0, r1 = rewards.get(agents[0], 0), rewards.get(agents[1], 0)
        if r0 > r1:
            return 1
        elif r0 < r1:
            return -1
        return 0

    def _create_agent(self, agent_config: AgentConfig) -> Agent:
        if isinstance(agent_config, RandomAgentConfig):
            return RandomAgent()

        if isinstance(agent_config, PolicyAgentConfig):
            cp_path = self._get_checkpoint_path(agent_config.model_name, agent_config.checkpoint)
            checkpoint = torch.load(cp_path, weights_only=False)
            backbone = self._create_backbone(checkpoint)
            label = f"{agent_config.model_name}@{agent_config.checkpoint}"
            return PolicyAgent(backbone, self.config.env.obs_space_format, name=label)

        if isinstance(agent_config, MCTSAgentConfig):
            from l2l_lab.agents import MCTSAgent
            from l2l_lab.utils.search import load_search_config

            cp_path = self._get_checkpoint_path(agent_config.model_name, agent_config.checkpoint)
            checkpoint = torch.load(cp_path, weights_only=False)
            backbone = self._create_backbone(checkpoint)
            search_config = load_search_config(agent_config.search_config_path)
            label = f"mcts[{agent_config.model_name}@{agent_config.checkpoint}]"
            return MCTSAgent(
                model=backbone,
                is_recurrent=agent_config.is_recurrent,
                search_config=search_config,
                obs_space_format=self.config.env.obs_space_format,
                name=label,
            )

        raise ValueError(f"Unknown agent config type: {type(agent_config)}")

    def _create_backbone(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        architecture = checkpoint["architecture"]
        network_kwargs = checkpoint.get("network_kwargs", {})
        input_shape = tuple(checkpoint["input_shape"])
        num_actions = checkpoint["num_actions"]

        network_class = NetworkConfig(architecture=architecture).get_network_class()

        if architecture in CONV_ARCHITECTURES:
            in_channels = input_shape[0]
            rows, cols = input_shape[1], input_shape[2]
            backbone = network_class(
                in_channels=in_channels,
                policy_channels=num_actions // (rows * cols),
                **network_kwargs,
            )
        elif architecture in MLP_ARCHITECTURES:
            backbone = network_class(out_features=num_actions, **network_kwargs)
            dummy = torch.zeros((1,) + input_shape, dtype=torch.float32)
            with torch.no_grad():
                _ = backbone(dummy)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        backbone.load_state_dict(checkpoint["state_dict"])
        backbone.eval()
        return backbone
