from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from l2l_lab.agents import PolicyAgent, RandomAgent
from l2l_lab.configs.training.EvaluationConfig import (CheckpointEvalEntry,
                                                       EvaluationConfig,
                                                       TrainingEvalEntry)
from l2l_lab.testing.tester import GameResults, Tester

if TYPE_CHECKING:
    from l2l_lab.agents.agent import Agent
    from l2l_lab.backends.backend_base import AlgorithmBackend
    from l2l_lab.configs.common.EnvConfig import EnvConfig
    from l2l_lab.configs.training.network import BaseNetworkConfig
    from l2l_lab.reporting import Reporter


EvalEntry = Union[TrainingEvalEntry, CheckpointEvalEntry]


class Evaluator:

    def __init__(
        self,
        eval_config: EvaluationConfig,
        backend: "AlgorithmBackend",
        env_config: "EnvConfig",
        network_config: "BaseNetworkConfig",
    ) -> None:
        self.eval_config = eval_config
        self.backend = backend
        self.env_config = env_config
        self.network_config = network_config
        self.reporter: Optional["Reporter"] = None

    def labels(self) -> list[str]:
        return self.eval_config.all_labels()

    def is_reporter_enabled(self) -> bool:
        return self.reporter is not None and self.reporter.enabled

    def label_to_type_map(self) -> Dict[str, str]:
        """{label -> 'training' | 'checkpoint'} for every configured entry."""
        mapping: Dict[str, str] = {}
        for entry in self.eval_config.training_eval:
            mapping[entry.label] = "training"
        for entry in self.eval_config.checkpoint_eval:
            mapping[entry.label] = "checkpoint"
        return mapping

    def run_training_evals(self, iteration: int) -> Dict[str, Optional[GameResults]]:
        results: Dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.training_eval:
            if iteration % entry.interval != 0:
                results[entry.label] = None
                continue
            print(f"\nRunning training eval for iteration {iteration}...\n")
            player = self._build_player_agent(entry)
            opponent = self._build_baseline_opponent(entry.opponent, entry.search_config_path)
            results[entry.label] = self._play_balanced(
                player, opponent, entry.games_per_player,
                iteration=iteration, label=entry.label,
            )
        return results

    def run_checkpoint_evals(
        self,
        previous_checkpoint: Optional[Path],
        iteration: int = 0,
    ) -> Dict[str, Optional[GameResults]]:
        results: Dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.checkpoint_eval:
            needs_previous = entry.opponent in ("policy", "alphazero_mcts")
            if needs_previous and previous_checkpoint is None:
                results[entry.label] = None
                continue
            print(f"\nRunning checkpoint eval for iteration {iteration}...\n")
            player = self._build_player_agent(entry)
            opponent = self._build_opponent_agent(entry, previous_checkpoint)
            results[entry.label] = self._play_balanced(
                player, opponent, entry.games_per_player,
                iteration=iteration, label=entry.label,
            )
        return results

    def _build_player_agent(self, entry: EvalEntry) -> "Agent":
        if entry.player == "traditional_mcts":
            return self._build_traditional_mcts_agent(entry.search_config_path, name="current_traditional_mcts")
        model = self.backend.get_eval_model()
        return self._wrap_model(model, entry.player, entry.search_config_path, name_prefix="current")

    def _build_opponent_agent(
        self,
        entry: CheckpointEvalEntry,
        previous_checkpoint: Optional[Path],
    ) -> "Agent":
        if entry.opponent in ("random", "traditional_mcts"):
            return self._build_baseline_opponent(entry.opponent, entry.search_config_path)
        assert previous_checkpoint is not None
        model = self.backend.get_model_from_checkpoint(previous_checkpoint)
        return self._wrap_model(model, entry.opponent, entry.search_config_path, name_prefix="prev")

    def _build_baseline_opponent(
        self, opponent_type: str, search_config_path: Optional[str],
    ) -> "Agent":
        if opponent_type == "random":
            return RandomAgent()
        if opponent_type == "traditional_mcts":
            return self._build_traditional_mcts_agent(search_config_path, name="traditional_mcts")
        raise ValueError(f"Unsupported baseline opponent type: {opponent_type!r}")

    def _build_traditional_mcts_agent(
        self, search_config_path: Optional[str], name: str,
    ) -> "Agent":
        from l2l_lab.agents import TraditionalMCTSAgent
        from l2l_lab.utils.search import load_search_config

        search_config = load_search_config(search_config_path)
        return TraditionalMCTSAgent(
            search_config=search_config,
            obs_space_format=self.env_config.obs_space_format,
            name=name,
        )

    def _wrap_model(
        self,
        model,
        agent_type: str,
        search_config_path: Optional[str],
        name_prefix: str,
    ) -> "Agent":
        is_recurrent = self.network_config.is_recurrent()
        recurrent_iterations = (
            self.network_config.recurrent_iterations if is_recurrent else 1
        )
        if agent_type == "policy":
            return PolicyAgent(
                model,
                self.env_config.obs_space_format,
                is_recurrent=is_recurrent,
                recurrent_iterations=recurrent_iterations,
                name=f"{name_prefix}_policy",
            )
        if agent_type == "alphazero_mcts":
            from l2l_lab.agents import AlphaZeroMCTSAgent
            from l2l_lab.utils.search import load_search_config

            search_config = load_search_config(search_config_path)
            return AlphaZeroMCTSAgent(
                model=model,
                is_recurrent=is_recurrent,
                recurrent_iterations=recurrent_iterations,
                search_config=search_config,
                obs_space_format=self.env_config.obs_space_format,
                name=f"{name_prefix}_alphazero_mcts",
            )
        raise ValueError(f"Unsupported agent type for eval: {agent_type!r}")

    def _play_balanced(
        self,
        player: "Agent",
        opponent: "Agent",
        games_per_side: int,
        iteration: int = 0,
        label: str = "",
    ) -> GameResults:
        reports_to_capture = self.reporter.cfg.sample_games_per_eval if self.is_reporter_enabled() else 0

        as_p0 = Tester.play_games(
            p0=player, p1=opponent, env_config=self.env_config,
            num_games=games_per_side, reports_to_capture=reports_to_capture,
        )
        as_p1_raw = Tester.play_games(
            p0=opponent, p1=player, env_config=self.env_config,
            num_games=games_per_side, reports_to_capture=reports_to_capture,
        )

        if self.is_reporter_enabled():
            for report in as_p0.reports:
                self.reporter.add_game_report(iteration, label, "as_p0", report)
            for report in as_p1_raw.reports:
                self.reporter.add_game_report(iteration, label, "as_p1", report)

        as_p1 = GameResults(
            wins=as_p1_raw.losses,
            losses=as_p1_raw.wins,
            draws=as_p1_raw.draws,
            total=as_p1_raw.total,
            avg_moves=as_p1_raw.avg_moves,
            elapsed_time=as_p1_raw.elapsed_time,
        )

        total = as_p0.total + as_p1.total
        wins = as_p0.wins + as_p1.wins
        losses = as_p0.losses + as_p1.losses
        draws = as_p0.draws + as_p1.draws
        elapsed = as_p0.elapsed_time + as_p1.elapsed_time
        avg_moves = (
            (as_p0.avg_moves * as_p0.total + as_p1.avg_moves * as_p1.total) / total
            if total > 0 else 0.0
        )
        return GameResults(
            wins=wins,
            losses=losses,
            draws=draws,
            total=total,
            avg_moves=avg_moves,
            elapsed_time=elapsed,
            as_p0=as_p0,
            as_p1=as_p1,
        )
