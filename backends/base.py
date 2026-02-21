from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent
    from checkpoint_utils import CheckpointData
    from configs.definition.training.TrainingConfig import TrainingConfig


@dataclass
class StepResult:
    iteration: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class AlgorithmBackend(ABC):

    def __init__(self):
        self.metrics_queue: Queue[Optional[StepResult]] = Queue()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
        ...

    @abstractmethod
    def restore(self, config: TrainingConfig, model_dir: Path, checkpoint_dir: Path) -> Tuple[int, Optional[CheckpointData]]:
        """Restore from checkpoint. Return (start_iteration, checkpoint_data)."""
        ...

    @abstractmethod
    def start_training(self, start_iteration: int, total_iterations: int) -> None:
        """Launch training in a background thread. Push StepResult to metrics_queue
        after each step. Push None when done."""
        ...

    @abstractmethod
    def create_eval_agent(self) -> Agent:
        """Snapshot current model into an Agent for evaluation (thread-safe read)."""
        ...

    @abstractmethod
    def create_agent_from_checkpoint(self, checkpoint_path: Path) -> Agent:
        """Load a previously saved checkpoint into an Agent for evaluation."""
        ...

    @abstractmethod
    def get_weight_parameters(self) -> Optional[Iterator]:
        """Return model parameters for weight stats collection (thread-safe read)."""
        ...

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: Path, iteration: int, metrics: Dict[str, List]) -> None:
        ...

    @abstractmethod
    def wait_for_completion(self) -> None:
        """Block until the training thread finishes."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources (stop Ray, free GPU, etc.)."""
        ...
