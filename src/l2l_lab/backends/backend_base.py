from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from l2l_lab.utils.checkpoint import delete_checkpoint_dirs_past

if TYPE_CHECKING:
    import torch

    from l2l_lab.configs.training.TrainingConfig import TrainingConfig


@dataclass
class StepResult:
    iteration: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None


class AlgorithmBackend(ABC):

    def __init__(self):
        self.step_queue: Queue[Optional[StepResult]] = Queue()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        self._checkpoint_interval: int = 0
        self._checkpoint_base_dir: Optional[Path] = None
        self._start_iteration: int = 0
        self._total_iterations: int = 0

    def configure_checkpointing(self, interval: int, base_dir: Path) -> None:
        self._checkpoint_interval = interval
        self._checkpoint_base_dir = base_dir

    def request_stop(self) -> None:
        """Signal the training thread to stop after the current step."""
        self._stop_event.set()

    def wait_for_training(self, timeout: Optional[float] = 30) -> None:
        """Wait for the training thread to finish."""
        if self._training_thread is not None:
            self._training_thread.join(timeout=timeout)

    def start_training(self) -> None:
        """Launch `_train` in a background thread. The iteration range is read
        from the state stored during `setup`/`restore`."""
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._train,
            daemon=True,
        )
        self._training_thread.start()

    def delete_checkpoints_past(self, model_dir: Path, iteration: int) -> None:
        """Remove every checkpoint on disk whose iteration is greater than `iteration`."""
        delete_checkpoint_dirs_past(model_dir, iteration)

    def get_reporter_csv_keys(self) -> List[str]:
        """Return the step_metrics keys this backend wants written to the
        per-iteration CSV. Default is no CSV columns; override per backend."""
        return []

    def on_checkpoint_saved(self, model_dir: Path, iteration: int) -> None:
        """Post-save hook. Called by `Trainer` right after a checkpoint lands
        on disk. Default is a no-op."""
        pass

    def _print_step_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Per-step log. Called on the training thread at the end of each step.
        Default is a no-op; override in backends that want a step summary."""
        pass

    def _print_training_info(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """Periodic backend-specific log. Called on the training thread every
        `info_interval` steps. Default is a no-op."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def setup(self, config: TrainingConfig, model_dir: Path) -> None:
        ...

    @abstractmethod
    def restore(self, config: TrainingConfig, model_dir: Path) -> int:
        """Restore backend state from checkpoint. Return start_iteration."""
        ...

    @abstractmethod
    def get_eval_model(self) -> "torch.nn.Module":
        """Return an eval-mode snapshot of the currently-training model."""
        ...

    @abstractmethod
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> "torch.nn.Module":
        """Load and return an eval-mode model from a training checkpoint directory."""
        ...

    @abstractmethod
    def get_weight_parameters(self) -> Optional[Iterator]:
        ...

    @abstractmethod
    def save_final_checkpoint(self, iteration: int) -> Optional[Path]:
        """Capture and write a final checkpoint synchronously. Called after the
        training thread has been joined. Return the checkpoint directory, or
        None if checkpointing is disabled."""
        ...

    @abstractmethod
    def wait_for_pending_checkpoints(self) -> None:
        """Block until queued checkpoint writes have been flushed to disk."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    @abstractmethod
    def _train(self) -> None:
        """Run the training loop on the training thread. Push StepResult to
        `step_queue` after each step; push None when done."""
        ...
