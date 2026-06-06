from dataclasses import dataclass, field
from typing import Optional


PLAYER_TYPES = ("policy", "alphazero_mcts", "traditional_mcts")
OPPONENT_TYPES = ("random", "policy", "alphazero_mcts", "traditional_mcts")
TRAINING_EVAL_OPPONENT_TYPES = ("random", "traditional_mcts")
MCTS_PLAYER_TYPES = ("alphazero_mcts", "traditional_mcts")


@dataclass
class TrainingEvalEntry:
    player: str
    opponent: str
    games_per_player: int
    interval: int
    search_config_path: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.player}_vs_{self.opponent}"


@dataclass
class CheckpointEvalEntry:
    player: str
    opponent: str
    games_per_player: int
    search_config_path: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.player}_vs_{self.opponent}"


@dataclass
class EvaluationConfig:
    training_eval: list[TrainingEvalEntry] = field(default_factory=list)
    checkpoint_eval: list[CheckpointEvalEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        labels: list[str] = []

        for entry in self.training_eval:
            self._validate_player(entry.player)
            self._validate_training_eval_opponent(entry.opponent)
            self._validate_mcts_has_search_config(
                entry.player, entry.opponent, entry.search_config_path,
            )
            self._validate_positive(entry.games_per_player, "games_per_player")
            self._validate_positive(entry.interval, "interval")
            labels.append(entry.label)

        for entry in self.checkpoint_eval:
            self._validate_player(entry.player)
            self._validate_opponent(entry.opponent)
            self._validate_mcts_has_search_config(
                entry.player, entry.opponent, entry.search_config_path,
            )
            self._validate_positive(entry.games_per_player, "games_per_player")
            labels.append(entry.label)

        duplicates = {lbl for lbl in labels if labels.count(lbl) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate evaluation labels: {sorted(duplicates)}. Each (player, opponent) "
                f"combination may only appear once across training_eval and checkpoint_eval."
            )

    def all_labels(self) -> list[str]:
        return [e.label for e in self.training_eval] + [e.label for e in self.checkpoint_eval]

    @staticmethod
    def _validate_player(player: str) -> None:
        if player not in PLAYER_TYPES:
            raise ValueError(f"Invalid player type: {player!r}. Must be one of {PLAYER_TYPES}.")

    @staticmethod
    def _validate_opponent(opponent: str) -> None:
        if opponent not in OPPONENT_TYPES:
            raise ValueError(f"Invalid opponent type: {opponent!r}. Must be one of {OPPONENT_TYPES}.")

    @staticmethod
    def _validate_training_eval_opponent(opponent: str) -> None:
        if opponent not in TRAINING_EVAL_OPPONENT_TYPES:
            raise ValueError(
                f"Invalid training_eval opponent: {opponent!r}. "
                f"Must be one of {TRAINING_EVAL_OPPONENT_TYPES} "
                "(opponents that depend on a previous checkpoint belong in checkpoint_eval)."
            )

    @staticmethod
    def _validate_mcts_has_search_config(
        player: str, opponent: str, search_config_path: Optional[str],
    ) -> None:
        if (player in MCTS_PLAYER_TYPES or opponent in MCTS_PLAYER_TYPES) and not search_config_path:
            raise ValueError(
                "Evaluation entry with an mcts player or opponent requires a search_config_path."
            )

    @staticmethod
    def _validate_positive(value: int, field_name: str) -> None:
        if value <= 0:
            raise ValueError(f"Evaluation entry field {field_name!r} must be > 0, got {value}.")
