from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from l2l_lab.agents import PolicyAgent, RandomAgent
from l2l_lab.configs.training.EvaluationConfig import (
    CheckpointEvalEntry, EvaluationConfig, TrainingEvalEntry)
from l2l_lab.testing.tester import GameResults, Tester

if TYPE_CHECKING:
    from l2l_lab.agents.agent import Agent
    from l2l_lab.backends.base import AlgorithmBackend
    from l2l_lab.configs.common.EnvConfig import EnvConfig


EvalEntry = Union[TrainingEvalEntry, CheckpointEvalEntry]


class Evaluator:

    def __init__(
        self,
        eval_config: EvaluationConfig,
        backend: "AlgorithmBackend",
        env_config: "EnvConfig",
    ) -> None:
        self.eval_config = eval_config
        self.backend = backend
        self.env_config = env_config

    def labels(self) -> list[str]:
        return self.eval_config.all_labels()

    def run_training_evals(self, iteration: int) -> Dict[str, Optional[GameResults]]:
        results: Dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.training_eval:
            if iteration % entry.interval != 0:
                results[entry.label] = None
                continue
            player = self._build_player_agent(entry)
            opponent = RandomAgent()
            results[entry.label] = self._play_balanced(player, opponent, entry.games_per_player)
            self._restore_training_mode()
        return results

    def run_checkpoint_evals(
        self,
        previous_checkpoint: Optional[Path],
    ) -> Dict[str, Optional[GameResults]]:
        results: Dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.checkpoint_eval:
            needs_previous = entry.opponent in ("policy", "mcts")
            if needs_previous and previous_checkpoint is None:
                results[entry.label] = None
                continue
            player = self._build_player_agent(entry)
            opponent = self._build_opponent_agent(entry, previous_checkpoint)
            results[entry.label] = self._play_balanced(player, opponent, entry.games_per_player)
            self._restore_training_mode()
        return results

    def _build_player_agent(self, entry: EvalEntry) -> "Agent":
        model = self.backend.get_eval_model()
        return self._wrap_model(model, entry.player, entry.search_config_path, name_prefix="current")

    def _build_opponent_agent(
        self,
        entry: CheckpointEvalEntry,
        previous_checkpoint: Optional[Path],
    ) -> "Agent":
        if entry.opponent == "random":
            return RandomAgent()
        assert previous_checkpoint is not None
        model = self.backend.get_model_from_checkpoint(previous_checkpoint)
        return self._wrap_model(model, entry.opponent, entry.search_config_path, name_prefix="prev")

    def _wrap_model(
        self,
        model,
        agent_type: str,
        search_config_path: Optional[str],
        name_prefix: str,
    ) -> "Agent":
        if agent_type == "policy":
            return PolicyAgent(model, self.env_config.obs_space_format, name=f"{name_prefix}_policy")
        if agent_type == "mcts":
            from l2l_lab.agents import MCTSAgent
            from l2l_lab.utils.search import load_search_config

            search_config = load_search_config(search_config_path)
            # FIXME: is_recurrent should be threaded through from the trainer/network config.
            is_recurrent = False
            return MCTSAgent(
                model=model,
                is_recurrent=is_recurrent,
                search_config=search_config,
                obs_space_format=self.env_config.obs_space_format,
                name=f"{name_prefix}_mcts",
            )
        raise ValueError(f"Unsupported agent type for eval: {agent_type!r}")

    def _play_balanced(
        self,
        player: "Agent",
        opponent: "Agent",
        games_per_side: int,
    ) -> GameResults:
        as_p0 = Tester.play_games(p0=player, p1=opponent, env_config=self.env_config, num_games=games_per_side)
        as_p1_raw = Tester.play_games(p0=opponent, p1=player, env_config=self.env_config, num_games=games_per_side)

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

    def _restore_training_mode(self) -> None:
        # Some backends (e.g. RLlib) require flipping the module back to train mode
        # after we grab it for inference.
        if hasattr(self.backend, "_set_module_training"):
            self.backend._set_module_training()
