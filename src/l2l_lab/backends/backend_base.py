import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Iterator, Optional

from l2l_lab.utils.checkpoint import delete_checkpoint_dirs_past, list_checkpoint_iterations
from l2l_lab.utils.common import check_interval

if TYPE_CHECKING:
    import torch

    from l2l_lab.configs.training.training_config import TrainingConfig


logger = logging.getLogger("l2l_lab")


@dataclass
class StepResult:
    iteration: int
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None
    eval_model: Optional[torch.nn.Module] = None


class AlgorithmBackend(ABC):

    def __init__(self):
        self.step_queue: Queue[Optional[StepResult]] = Queue()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        self._checkpoint_interval: int = 0
        self._checkpoint_base_dir: Optional[Path] = None
        self._snapshot_intervals: list[int] = []
        self._starting_iteration: int = 0
        self._total_iterations: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def init(self) -> None:
        ...

    @abstractmethod
    def prepare(self, config: TrainingConfig) -> None:
        """Build the model and per-run state needed before either loading or a fresh start."""
        ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: Path) -> None:
        """Load backend state from `checkpoint_dir`; raise if it cannot be loaded."""
        ...

    @abstractmethod
    def init_fresh(self) -> None:
        """Initialize backend state for a run that loads no checkpoint."""
        ...

    @abstractmethod
    def get_eval_model(self) -> torch.nn.Module:
        """Return an eval-mode snapshot of the currently-training model."""
        ...

    @abstractmethod
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> torch.nn.Module:
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
    def train(self) -> None:
        """Run the training loop on the training thread. Push StepResult to
        `step_queue` after each step; push None when done."""
        ...

    def get_reporter_csv_keys(self) -> list[str]:
        """Return the step_metrics keys this backend wants written to the
        per-iteration CSV. Default is no CSV columns; override per backend."""
        return []

    def on_checkpoint_saved(self, model_dir: Path, iteration: int) -> None:
        """Post-save hook. Called by `Trainer` right after a checkpoint lands
        on disk. Default is a no-op."""
        pass

    def print_step_info(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Per-step log. Called on the training thread at the end of each step.
        Default is a no-op; override in backends that want a step summary."""
        pass

    def print_training_info(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Periodic backend-specific log. Called on the training thread every
        `info_interval` steps. Default is a no-op."""
        pass

    def configure_checkpointing(
        self, base_dir: Path, checkpoint_interval: int, eval_intervals: list[int]
    ) -> None:
        """Set up checkpoint writing and the cadence at which the training thread
        must snapshot the model. A snapshot is captured on any step that a
        checkpoint or a training eval will consume, so the consumer can build
        eval agents from the weights as they were at that step."""
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_base_dir = base_dir
        self._snapshot_intervals = sorted({checkpoint_interval, *eval_intervals})

    def request_stop(self) -> None:
        """Signal the training thread to stop after the current step."""
        self._stop_event.set()

    def wait_for_training(self, timeout: Optional[float] = 120) -> None:
        """Wait for the training thread to gracefully finish, up to `timeout` seconds."""
        if self._training_thread is None:
            return
        if timeout is not None:
            logger.info(f"\nWaiting up to {int(timeout)}s for the training backend to finish...\n")
        self._training_thread.join(timeout=timeout)
        if self._training_thread.is_alive():
            logger.warning(
                f"WARNING: training backend still running after {int(timeout)}s; shutdown may be incomplete."
            )

    def start_training(self) -> None:
        """Launch `train` in a background thread."""
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self.train,
            daemon=True,
        )
        self._training_thread.start()

    def delete_checkpoints_past(self, model_dir: Path, iteration: int) -> None:
        """Remove every checkpoint on disk whose iteration is greater than `iteration`."""
        delete_checkpoint_dirs_past(model_dir, iteration)

    def new_run(self, config: TrainingConfig, model_dir: Path) -> None:
        self.prepare(config)
        self.init_fresh()
        self._starting_iteration = 0

    def restore_run(self, config: TrainingConfig, model_dir: Path) -> int:
        self.prepare(config)
        loaded_iteration = self._load_latest_loadable_checkpoint(
            model_dir, config.backend.continue_from_iteration
        )
        self._starting_iteration = loaded_iteration + 1
        return loaded_iteration

    def _load_latest_loadable_checkpoint(self, model_dir: Path, target_iteration: Optional[int]) -> int:
        """Load the highest-numbered checkpoint at or below `target_iteration`, walking to
        earlier ones when a load fails, and return its iteration. Start fresh (returning 0)
        when nothing loads. No `target_iteration` considers every checkpoint."""
        candidates = list_checkpoint_iterations(model_dir)
        if target_iteration is not None:
            candidates = [iteration for iteration in candidates if iteration <= target_iteration]

        for iteration in sorted(candidates, reverse=True):
            checkpoint_dir = model_dir / "checkpoints" / str(iteration)
            try:
                self.load_checkpoint(checkpoint_dir)
            except Exception as exc:
                logger.warning(f"Checkpoint {iteration} could not be loaded ({exc}); trying an earlier one.")
                continue
            logger.info(f"Restored from checkpoint {iteration}.")
            return iteration

        logger.info("No loadable checkpoint found. Starting fresh.")
        self.init_fresh()
        return 0

    def _needs_snapshot(self, iterations_completed: int) -> bool:
        """True when an eval or checkpoint at this point will consume a model
        snapshot. Evaluated on the training thread to decide whether to capture
        the current weights for this step's `StepResult`."""
        return any(check_interval(iterations_completed, interval) for interval in self._snapshot_intervals)
