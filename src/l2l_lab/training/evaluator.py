from pathlib import Path
from typing import TYPE_CHECKING, Optional

from l2l_lab.agents import PolicyAgent, RandomAgent
from l2l_lab.configs.training.evaluation_config import EvaluationConfig
from l2l_lab.testing.tester import GameResults, Tester
from l2l_lab.utils.common import check_interval
from l2l_lab.utils.search import load_search_config
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from torch import nn

    from l2l_lab.agents.agent import Agent
    from l2l_lab.backends.backend_base import AlgorithmBackend
    from l2l_lab.configs.common.env_config import EnvConfig
    from l2l_lab.configs.training.network import BaseNetworkConfig


class Evaluator:

    def __init__(
        self,
        eval_config: EvaluationConfig,
        backend: AlgorithmBackend,
        env_config: EnvConfig,
        network_config: BaseNetworkConfig,
        reports_to_capture: int = 0,
    ) -> None:
        self.eval_config = eval_config
        self.backend = backend
        self.env_config = env_config
        self.network_config = network_config
        self.reports_to_capture = reports_to_capture

    def labels(self) -> list[str]:
        return self.eval_config.all_labels()

    def training_evals_due(self, iterations_completed: int) -> bool:
        """True when at least one `training_eval` entry fires at this point,
        so the caller knows whether an eval request is worth enqueuing."""
        return any(
            check_interval(iterations_completed, entry.interval) for entry in self.eval_config.training_eval
        )

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
        self, iteration: int, eval_model: Optional[nn.Module]
    ) -> dict[str, Optional[GameResults]]:
        logger.info("")
        iterations_completed = iteration + 1
        results: dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.training_eval:
            if not check_interval(iterations_completed, entry.interval):
                results[entry.label] = None
                continue
            logger.info(f"Starting training eval [{entry.player} x {entry.opponent}] for iteration {iteration}...")
            player = self._build_agent(entry.player, entry.search_config_path, eval_model, f"current_{entry.player}")
            opponent = self._build_agent(entry.opponent, entry.search_config_path, None, entry.opponent)
            results[entry.label] = self._play_balanced(player, opponent, entry.games_per_player)
            logger.info(f"Finished training eval [{entry.player} x {entry.opponent}]\n")
        return results

    def run_checkpoint_evals(
        self,
        previous_checkpoint: Optional[Path],
        iteration: int = 0,
        eval_model: Optional[nn.Module] = None,
    ) -> dict[str, Optional[GameResults]]:
        logger.info("")
        results: dict[str, Optional[GameResults]] = {}
        for entry in self.eval_config.checkpoint_eval:
            logger.info(f"Starting checkpoint eval [{entry.player} x {entry.opponent}] for iteration {iteration}...")
            opponent_model: Optional[nn.Module] = None
            if self._needs_model(entry.opponent):
                if previous_checkpoint is None:
                    logger.error(f"Skipping current checkpoint eval: no checkpoint model provided for {entry.opponent} opponent")
                    results[entry.label] = None
                    continue
                opponent_model = self.backend.get_model_from_checkpoint(previous_checkpoint)

            player = self._build_agent(entry.player, entry.search_config_path, eval_model, f"current_{entry.player}")
            opponent = self._build_agent(entry.opponent, entry.search_config_path, opponent_model, f"prev_{entry.opponent}")
            results[entry.label] = self._play_balanced(player, opponent, entry.games_per_player)
            logger.info(f"Finished checkpoint eval [{entry.player} x {entry.opponent}]\n")
        return results

    def _build_agent(
        self,
        agent_type: str,
        search_config_path: Optional[str],
        model: Optional[nn.Module],
        name: str,
    ) -> Agent:
        match agent_type:
            case "random":
                return RandomAgent(name=name)
            case "traditional_mcts":
                return self._build_traditional_mcts_agent(search_config_path, name)
            case "policy":
                return self._build_policy_agent(model, name)
            case "alphazero_mcts":
                return self._build_alphazero_mcts_agent(model, search_config_path, name)
            case _:
                raise ValueError(f"Unsupported agent type for eval: {agent_type!r}")

    def _build_policy_agent(self, model: nn.Module, name: str) -> PolicyAgent:
        return PolicyAgent(
            model,
            self.env_config.obs_space_format,
            is_recurrent=self.network_config.is_recurrent(),
            recurrent_iterations=self._recurrent_iterations(),
            name=name,
        )

    def _build_alphazero_mcts_agent(
        self, model: nn.Module, search_config_path: Optional[str], name: str
    ) -> Agent:
        from l2l_lab.agents import AlphaZeroMCTSAgent

        return AlphaZeroMCTSAgent(
            model=model,
            is_recurrent=self.network_config.is_recurrent(),
            recurrent_iterations=self._recurrent_iterations(),
            search_config=load_search_config(search_config_path),
            obs_space_format=self.env_config.obs_space_format,
            name=name,
        )

    def _build_traditional_mcts_agent(self, search_config_path: Optional[str], name: str) -> Agent:
        from l2l_lab.agents import TraditionalMCTSAgent

        return TraditionalMCTSAgent(
            search_config=load_search_config(search_config_path),
            obs_space_format=self.env_config.obs_space_format,
            name=name,
        )

    def _play_balanced(
        self,
        player: Agent,
        opponent: Agent,
        games_per_side: int,
    ) -> GameResults:
        as_p0 = Tester.play_games(
            p0=player, p1=opponent, env_config=self.env_config,
            num_games=games_per_side, reports_to_capture=self.reports_to_capture,
        )
        as_p1_raw = Tester.play_games(
            p0=opponent, p1=player, env_config=self.env_config,
            num_games=games_per_side, reports_to_capture=self.reports_to_capture,
        )

        as_p1 = GameResults(
            wins=as_p1_raw.losses,
            losses=as_p1_raw.wins,
            draws=as_p1_raw.draws,
            total=as_p1_raw.total,
            avg_moves=as_p1_raw.avg_moves,
            elapsed_time=as_p1_raw.elapsed_time,
            reports=as_p1_raw.reports,
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
    
    def _needs_model(self, agent_type: str) -> bool:
        return agent_type in ("policy", "alphazero_mcts")
    
    def _recurrent_iterations(self) -> int:
        if self.network_config.is_recurrent():
            return self.network_config.recurrent_iterations
        return 1
