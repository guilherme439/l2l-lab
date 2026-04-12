from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from agents import Agent, PolicyAgent, RandomAgent
from backends.obs_utils import obs_to_state_provider
from configs.definition.common.EnvConfig import EnvConfig
from configs.definition.testing.agents.AgentConfig import AgentConfig
from configs.definition.testing.agents.PolicyAgentConfig import \
    PolicyAgentConfig
from configs.definition.testing.agents.RandomAgentConfig import \
    RandomAgentConfig
from configs.definition.testing.TestingConfig import TestingConfig
from configs.definition.training.NetworkConfig import (CONV_ARCHITECTURES,
                                                       MLP_ARCHITECTURES,
                                                       NetworkConfig)
from envs.registry import create_env

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
    def evaluate_agent_vs_random(agent: Agent, env_config: EnvConfig, num_games: int = 10) -> GameResults:
        opponent = RandomAgent()
        return Tester._run_games(
            env_config, num_games,
            agent_p0=agent,
            agent_p1=opponent,
            alternate_positions=True,
        )

    @staticmethod
    def evaluate_agent_vs_agent(
        agent: Agent,
        opponent: Agent,
        env_config: EnvConfig,
        num_games: int = 10,
    ) -> GameResults:
        return Tester._run_games(
            env_config, num_games,
            agent_p0=agent,
            agent_p1=opponent,
            alternate_positions=True,
        )

    @staticmethod
    def evaluate_agent_vs_checkpoint(
        agent: Agent,
        checkpoint_path: Path,
        env_config: EnvConfig,
        num_games: int = 10,
    ) -> GameResults:
        if not checkpoint_path.exists():
            return GameResults(0, 0, 0, 0)

        opponent_backbone = Tester._load_backbone_from_checkpoint(checkpoint_path)
        opponent = PolicyAgent(opponent_backbone, name="checkpoint")

        return Tester._run_games(
            env_config, num_games,
            agent_p0=agent,
            agent_p1=opponent,
            alternate_positions=True,
        )

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

        results = Tester._run_games(
            self.config.env, self.config.num_games,
            agent_p0=agent_p1,
            agent_p1=agent_p2,
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

    @staticmethod
    def _load_backbone_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
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

    @staticmethod
    def _run_games(
        env_config: EnvConfig,
        num_games: int,
        agent_p0: Agent,
        agent_p1: Agent,
        alternate_positions: bool = False,
    ) -> GameResults:
        start_time = time.time()
        wins, losses, draws = 0, 0, 0
        total_moves = 0
        p0_w, p0_l, p0_d, p0_n, p0_m = 0, 0, 0, 0, 0
        p1_w, p1_l, p1_d, p1_n, p1_m = 0, 0, 0, 0, 0
        env = create_env(env_config.name, **env_config.kwargs)
        obs_to_state = obs_to_state_provider(env_config.obs_space_format)

        for game_idx in range(num_games):
            swapped = alternate_positions and (game_idx % 2 == 1)
            current_p0 = agent_p1 if swapped else agent_p0
            current_p1 = agent_p0 if swapped else agent_p1

            env.reset()
            moves = 0

            while not Tester._is_env_done(env):
                agent_id = env.agent_selection
                obs = env.observe(agent_id)
                action_mask = obs["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]

                if len(valid_actions) == 0:
                    print("\nno valid actions found")
                    env.step(None)
                    continue

                state = obs_to_state(obs, agent_id)

                if agent_id == "player_0":
                    action = current_p0.choose_action(state, action_mask)
                else:
                    action = current_p1.choose_action(state, action_mask)

                env.step(action)
                if action is not None:
                    moves += 1

            result = Tester._get_game_result(env)
            if swapped:
                result = -result
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

            total_moves += moves

            if alternate_positions:
                if swapped:
                    p1_n += 1; p1_m += moves
                    if result > 0: p1_w += 1
                    elif result < 0: p1_l += 1
                    else: p1_d += 1
                else:
                    p0_n += 1; p0_m += moves
                    if result > 0: p0_w += 1
                    elif result < 0: p0_l += 1
                    else: p0_d += 1

        elapsed = time.time() - start_time
        avg_moves = (total_moves / num_games) if num_games > 0 else 0.0

        as_p0, as_p1 = None, None
        if alternate_positions and p0_n > 0 and p1_n > 0:
            as_p0 = GameResults(p0_w, p0_l, p0_d, p0_n, p0_m / p0_n)
            as_p1 = GameResults(p1_w, p1_l, p1_d, p1_n, p1_m / p1_n)

        return GameResults(wins, losses, draws, num_games, avg_moves, elapsed, as_p0, as_p1)

    def _create_agent(self, agent_config: AgentConfig) -> Agent:
        if isinstance(agent_config, RandomAgentConfig):
            return RandomAgent()

        if isinstance(agent_config, PolicyAgentConfig):
            cp_path = self._get_checkpoint_path(agent_config.model_name, agent_config.checkpoint)
            checkpoint = torch.load(cp_path, weights_only=False)
            backbone = self._create_backbone(checkpoint)
            label = f"{agent_config.model_name}@{agent_config.checkpoint}"
            return PolicyAgent(backbone, name=label)

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
