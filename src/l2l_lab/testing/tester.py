from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from l2l_lab.agents import Agent, PolicyAgent, RandomAgent
from l2l_lab.configs.common.EnvConfig import EnvConfig
from l2l_lab.configs.testing.agents.AgentConfig import AgentConfig
from l2l_lab.configs.testing.agents.AlphaZeroMCTSAgentConfig import AlphaZeroMCTSAgentConfig
from l2l_lab.configs.testing.agents.PolicyAgentConfig import \
    PolicyAgentConfig
from l2l_lab.configs.testing.agents.RandomAgentConfig import \
    RandomAgentConfig
from l2l_lab.configs.testing.agents.TraditionalMCTSAgentConfig import TraditionalMCTSAgentConfig
from l2l_lab.configs.testing.TestingConfig import TestingConfig
from l2l_lab.envs.registry import create_env
from l2l_lab.reporting.types import GameReport
from l2l_lab.utils.checkpoint import load_checkpoint_file, load_model_state_dict
from l2l_lab.utils.common import clone_observation

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
    reports: list[GameReport] = field(default_factory=list)

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
        reports_to_capture: int = 0,
    ) -> GameResults:
        """
            Play `num_games` games with a fixed (p0, p1) pair.
            Returns results from p0's perspective.

            The first ``reports_to_capture`` games have their move sequences
            recorded as ``GameReport`` objects and returned in
            ``GameResults.reports``.
        """
        start_time = time.time()
        wins, losses, draws = 0, 0, 0
        total_moves = 0
        env = create_env(env_config.name, **env_config.kwargs)

        captured_reports: list[GameReport] = []

        for _ in range(num_games):
            env.reset()
            moves = 0
            capture = len(captured_reports) < reports_to_capture
            recorded_moves: list[tuple[str, Optional[int], Dict[str, Any]]] = []

            while not Tester._is_env_done(env):
                agent_id = env.agent_selection
                obs = env.observe(agent_id)
                action_mask = obs["action_mask"]

                if np.sum(action_mask) == 0:
                    print("\nno valid actions found")
                    env.step(None)
                    continue

                current = p0 if agent_id == "player_0" else p1
                action = current.choose_action(env)

                if capture:
                    recorded_moves.append((agent_id, action, clone_observation(obs)))

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

            if capture:
                captured_reports.append(GameReport(
                    p0_name=getattr(p0, "name", p0.__class__.__name__),
                    p1_name=getattr(p1, "name", p1.__class__.__name__),
                    result_from_p0=result,
                    num_moves=moves,
                    moves=recorded_moves,
                ))

        elapsed = time.time() - start_time
        avg_moves = (total_moves / num_games) if num_games > 0 else 0.0
        return GameResults(wins, losses, draws, num_games, avg_moves, elapsed, reports=captured_reports)

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
    def _get_checkpoint_dir(model_name: str, checkpoint: int) -> Path:
        model_dir = MODELS_DIR / f"{model_name}"
        return model_dir / "checkpoints" / str(checkpoint)

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
            cp_dir = self._get_checkpoint_dir(agent_config.model_name, agent_config.checkpoint)
            backbone = self._create_backbone(cp_dir)
            label = f"{agent_config.model_name}@{agent_config.checkpoint}"
            return PolicyAgent(backbone, self.config.env.obs_space_format, name=label)

        if isinstance(agent_config, AlphaZeroMCTSAgentConfig):
            from l2l_lab.agents import AlphaZeroMCTSAgent
            from l2l_lab.utils.search import load_search_config

            cp_dir = self._get_checkpoint_dir(agent_config.model_name, agent_config.checkpoint)
            backbone = self._create_backbone(cp_dir)
            search_config = load_search_config(agent_config.search_config_path)
            label = f"alphazero_mcts[{agent_config.model_name}@{agent_config.checkpoint}]"
            return AlphaZeroMCTSAgent(
                model=backbone,
                is_recurrent=agent_config.is_recurrent,
                search_config=search_config,
                obs_space_format=self.config.env.obs_space_format,
                name=label,
            )

        if isinstance(agent_config, TraditionalMCTSAgentConfig):
            from l2l_lab.agents import TraditionalMCTSAgent
            from l2l_lab.utils.search import load_search_config

            search_config = load_search_config(agent_config.search_config_path)
            return TraditionalMCTSAgent(
                search_config=search_config,
                obs_space_format=self.config.env.obs_space_format,
                name="traditional_mcts",
            )

        raise ValueError(f"Unknown agent config type: {type(agent_config)}")

    def _create_backbone(self, checkpoint_dir: Path) -> torch.nn.Module:
        model_dir = checkpoint_dir / "model"
        backbone = torch.load(model_dir / "base_class.pkl", weights_only=False)
        state_dict = load_checkpoint_file(model_dir / "weights.cp")
        load_model_state_dict(backbone, state_dict)
        backbone.eval()
        return backbone
