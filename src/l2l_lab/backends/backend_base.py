from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from l2l_lab.configs.training.TrainingConfig import TrainingConfig
    from l2l_lab.utils.checkpoint import CheckpointData


@dataclass
class StepResult:
    iteration: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class AlgorithmBackend(ABC):

    def __init__(self):
        self.step_queue: Queue[Optional[StepResult]] = Queue()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None

    def request_stop(self) -> None:
        """Signal the training thread to stop after the current step."""
        self._stop_event.set()

    def wait_for_training(self, timeout: Optional[float] = 30) -> None:
        """Wait for the training thread to finish."""
        if self._training_thread is not None:
            self._training_thread.join(timeout=timeout)

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

    def start_training(self, start_iteration: int, total_iterations: int) -> None:
        """Launch `_train` in a background thread."""
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._train,
            args=(start_iteration, total_iterations),
            daemon=True,
        )
        self._training_thread.start()

    @abstractmethod
    def _train(self, start_iteration: int, total_iterations: int) -> None:
        """Run the training loop on the training thread. Push StepResult to
        `step_queue` after each step; push None when done."""
        ...

    def _print_step_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Per-step log. Called on the training thread at the end of each step.
        Default is a no-op; override in backends that want a step summary."""
        pass

    def _print_training_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Periodic backend-specific log. Called on the training thread every
        `info_interval` steps. Default is a no-op."""
        pass

    def on_checkpoint_saved(self, model_dir: Path, iteration: int) -> None:
        """Post-save hook. Called by `Trainer` right after a checkpoint lands
        on disk. Default is a no-op."""
        pass

    @abstractmethod
    def get_eval_model(self) -> "torch.nn.Module":
        """Return an eval-mode snapshot of the currently-training model."""
        ...

    @abstractmethod
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> "torch.nn.Module":
        """Load and return an eval-mode model from a training checkpoint directory."""
        ...

    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return current training state for checkpointing."""
        ...

    @abstractmethod
    def get_weight_parameters(self) -> Optional[Iterator]:
        ...

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: Path, iteration: int,
                        metrics: Dict[str, List], checkpoint_data: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...
