from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


PLAYER_TYPES = ("policy", "mcts")
OPPONENT_TYPES = ("random", "policy", "mcts")


@dataclass
class TrainingEvalEntry:
    player: str
    games_per_player: int
    interval: int
    search_config_path: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.player}_vs_random"


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
    training_eval: List[TrainingEvalEntry] = field(default_factory=list)
    checkpoint_eval: List[CheckpointEvalEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        labels: List[str] = []

        for entry in self.training_eval:
            self._validate_player(entry.player)
            self._validate_mcts_has_search_config(entry.player, entry.search_config_path)
            self._validate_positive(entry.games_per_player, "games_per_player")
            self._validate_positive(entry.interval, "interval")
            labels.append(entry.label)

        for entry in self.checkpoint_eval:
            self._validate_player(entry.player)
            self._validate_opponent(entry.opponent)
            self._validate_mcts_has_search_config(entry.player, entry.search_config_path)
            self._validate_positive(entry.games_per_player, "games_per_player")
            if entry.opponent == "mcts" and entry.search_config_path is None:
                raise ValueError(
                    "checkpoint_eval entry with opponent='mcts' requires a search_config_path"
                )
            labels.append(entry.label)

        duplicates = {lbl for lbl in labels if labels.count(lbl) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate evaluation labels: {sorted(duplicates)}. Each (player, opponent) "
                f"combination may only appear once across training_eval and checkpoint_eval."
            )

    def all_labels(self) -> List[str]:
        return [e.label for e in self.training_eval] + [e.label for e in self.checkpoint_eval]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        training = [
            TrainingEvalEntry(
                player=item["player"],
                games_per_player=item["games_per_player"],
                interval=item["interval"],
                search_config_path=item.get("search_config_path"),
            )
            for item in data.get("training_eval", [])
        ]
        checkpoint = [
            CheckpointEvalEntry(
                player=item["player"],
                opponent=item["opponent"],
                games_per_player=item["games_per_player"],
                search_config_path=item.get("search_config_path"),
            )
            for item in data.get("checkpoint_eval", [])
        ]
        return cls(training_eval=training, checkpoint_eval=checkpoint)

    @staticmethod
    def _validate_player(player: str) -> None:
        if player not in PLAYER_TYPES:
            raise ValueError(f"Invalid player type: {player!r}. Must be one of {PLAYER_TYPES}.")

    @staticmethod
    def _validate_opponent(opponent: str) -> None:
        if opponent not in OPPONENT_TYPES:
            raise ValueError(f"Invalid opponent type: {opponent!r}. Must be one of {OPPONENT_TYPES}.")

    @staticmethod
    def _validate_mcts_has_search_config(player: str, search_config_path: Optional[str]) -> None:
        if player == "mcts" and not search_config_path:
            raise ValueError("Evaluation entry with player='mcts' requires a search_config_path.")

    @staticmethod
    def _validate_positive(value: int, field_name: str) -> None:
        if value <= 0:
            raise ValueError(f"Evaluation entry field {field_name!r} must be > 0, got {value}.")
