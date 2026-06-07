from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from l2l_lab.agents import PolicyAgent, RandomAgent
from l2l_lab.configs.training.evaluation_config import EvaluationConfig
from l2l_lab.testing.tester import GameResults, Tester
from l2l_lab.utils.common import check_interval
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    import torch

    from l2l_lab.agents.agent import Agent
    from l2l_lab.backends.backend_base import AlgorithmBackend
    from l2l_lab.configs.common.env_config import EnvConfig
    from l2l_lab.configs.training.network import BaseNetworkConfig
    from l2l_lab.reporting import Reporter


class Evaluator:

    def __init__(
        self,
        eval_config: EvaluationConfig,
        backend: AlgorithmBackend,
        env_config: EnvConfig,
        network_config: BaseNetworkConfig,
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

    def label_to_type_map(self) -> dict[str, str]:
        """{label -> 'training' | 'checkpoint'} for every configured entry."""
        mapping: dict[str, str] = {}
        for entry in self.eval_config.training_eval:
            mapping[entry.label] = "training"
        for entry in self.eval_config.checkpoint_eval:
            mapping[entry.label] = "checkpoint"
        return mapping

    def training_eval_intervals(self) -> list[int]:
        """Intervals at which training evals run, for the backend to schedule
        the weight snapshots those evals consume."""
        return [entry.interval for entry in self.eval_config.training_eval]

    def run_training_evals(
        self, iteration: int, eval_model: Optional["torch.nn.Module"]
    ) -> dict[str, Optional[GameResults]]:
        logger.info("")
        current_model = self._model_provider(eval_model)
        results: dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.training_eval:
            if not check_interval(iteration, entry.interval):
                results[entry.label] = None
                continue
            logger.info(f"Starting training eval [{entry.player} x {entry.opponent}] for iteration {iteration}...")
            player = self._build_agent(
                entry.player, current_model, entry.search_config_path, f"current_{entry.player}"
            )
            opponent = self._build_agent(
                entry.opponent, self._no_model, entry.search_config_path, entry.opponent
            )
            results[entry.label] = self._play_balanced(
                player, opponent, entry.games_per_player,
                iteration=iteration, label=entry.label,
            )
            logger.info(f"Finished training eval [{entry.player} x {entry.opponent}]\n")
        return results

    def run_checkpoint_evals(
        self,
        previous_checkpoint: Optional[Path],
        iteration: int = 0,
        eval_model: Optional["torch.nn.Module"] = None,
    ) -> dict[str, Optional[GameResults]]:
        logger.info("")
        current_model = self._model_provider(eval_model)

        def previous_model() -> torch.nn.Module:
            assert previous_checkpoint is not None
            return self.backend.get_model_from_checkpoint(previous_checkpoint)

        results: dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.checkpoint_eval:
            needs_previous = entry.opponent in ("policy", "alphazero_mcts")
            if needs_previous and previous_checkpoint is None:
                results[entry.label] = None
                continue
            logger.info(f"Starting checkpoint eval [{entry.player} x {entry.opponent}] for iteration {iteration}...")
            player = self._build_agent(
                entry.player, current_model, entry.search_config_path, f"current_{entry.player}"
            )
            opponent_name = f"prev_{entry.opponent}" if needs_previous else entry.opponent
            opponent = self._build_agent(
                entry.opponent, previous_model, entry.search_config_path, opponent_name
            )
            results[entry.label] = self._play_balanced(
                player, opponent, entry.games_per_player,
                iteration=iteration, label=entry.label,
            )
            logger.info(f"Finished checkpoint eval [{entry.player} x {entry.opponent}]\n")
        return results

    @staticmethod
    def _no_model() -> torch.nn.Module:
        raise RuntimeError("A model-backed agent was requested for an eval role that supplies no model.")

    @staticmethod
    def _model_provider(model: Optional["torch.nn.Module"]) -> Callable[[], "torch.nn.Module"]:
        """Wrap an optional in-memory model as a provider that fails loudly if the
        model is missing when a model-backed agent actually needs it."""
        def provide() -> torch.nn.Module:
            if model is None:
                raise RuntimeError(
                    "Eval requested the current model, but no weight snapshot was captured for this step."
                )
            return model
        return provide

    def _build_agent(
        self,
        agent_type: str,
        model_provider: Callable[[], "torch.nn.Module"],
        search_config_path: Optional[str],
        name: str,
    ) -> Agent:
        is_recurrent = self.network_config.is_recurrent()
        recurrent_iterations = self.network_config.recurrent_iterations if is_recurrent else 1
        match agent_type:
            case "random":
                return RandomAgent()
            case "traditional_mcts":
                from l2l_lab.agents import TraditionalMCTSAgent
                from l2l_lab.utils.search import load_search_config

                search_config = load_search_config(search_config_path)
                return TraditionalMCTSAgent(
                    search_config=search_config,
                    obs_space_format=self.env_config.obs_space_format,
                    name=name,
                )
            case "policy":
                return PolicyAgent(
                    model_provider(),
                    self.env_config.obs_space_format,
                    is_recurrent=is_recurrent,
                    recurrent_iterations=recurrent_iterations,
                    name=name,
                )
            case "alphazero_mcts":
                from l2l_lab.agents import AlphaZeroMCTSAgent
                from l2l_lab.utils.search import load_search_config

                search_config = load_search_config(search_config_path)
                return AlphaZeroMCTSAgent(
                    model=model_provider(),
                    is_recurrent=is_recurrent,
                    recurrent_iterations=recurrent_iterations,
                    search_config=search_config,
                    obs_space_format=self.env_config.obs_space_format,
                    name=name,
                )
            case _:
                raise ValueError(f"Unsupported agent type for eval: {agent_type!r}")

    def _play_balanced(
        self,
        player: Agent,
        opponent: Agent,
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
