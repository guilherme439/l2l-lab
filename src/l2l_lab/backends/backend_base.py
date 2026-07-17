import logging
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Iterator, Optional

import torch

from l2l_lab._utils.checkpoint import CheckpointUtils
from l2l_lab._utils.common import CommonUtils

if TYPE_CHECKING:
    from torch import nn

    from l2l_lab.configs.training.training_config import TrainingConfig


logger = logging.getLogger("l2l_lab")


@dataclass
class StepResult:
    iteration: int
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None
    eval_model: Optional[nn.Module] = None


class AlgorithmBackend(ABC):

    def __init__(self):
        self.step_queue: Queue[Optional[StepResult]] = Queue()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        self._checkpoint_interval: int = 0
        self._checkpoint_base_dir: Optional[Path] = None
        self._network_template_path: Optional[Path] = None
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
    def _get_live_model(self) -> nn.Module:
        """Return the model as the backend currently holds it, with no copying.
        Only safe to call from the training thread while it owns the model."""
        ...

    @abstractmethod
    def get_model_from_checkpoint(self, checkpoint_dir: Path) -> nn.Module:
        """Load and return an eval-mode model from a training checkpoint directory."""
        ...

    @abstractmethod
    def get_weight_parameters(self) -> Optional[Iterator]:
        ...

    @abstractmethod
    def save_final_checkpoint(self, iteration: int) -> Optional[Path]:
        """Write a final checkpoint. Called after the training thread has been
        joined. Return the checkpoint directory, or None if checkpointing is
        disabled."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        """Run the training loop on the training thread. Push StepResult to
        `step_queue` after each step; push None when done."""
        ...

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
        self, base_dir: Path, checkpoint_interval: int, eval_intervals: list[int], report_interval: int
    ) -> None:
        """Set up checkpoint writing and the cadence at which the training thread
        must snapshot the model. A snapshot is captured on any step that a
        checkpoint, a training eval, or a reporting snapshot will consume, so the
        consumer can build eval agents (or a report) from the weights as they
        were at that step."""
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_base_dir = base_dir
        self._snapshot_intervals = sorted({checkpoint_interval, report_interval, *eval_intervals})
        if checkpoint_interval > 0:
            self._network_template_path = base_dir / "network_template.pkl"
            self._write_network_template(self._network_template_path)

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
        CheckpointUtils.delete_checkpoint_dirs_past(model_dir, iteration)

    def new_run(self, config: TrainingConfig, model_dir: Path) -> None:
        self.prepare(config)
        self.init_fresh()
        self._starting_iteration = 0
        self._log_network_summary()

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
        candidates = CheckpointUtils.list_checkpoint_iterations(model_dir)
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
        """True when an eval, checkpoint, or report at this point will consume a
        model snapshot. Evaluated on the training thread to decide whether to
        capture the current weights for this step's `StepResult`."""
        return any(CommonUtils.check_interval(iterations_completed, interval) for interval in self._snapshot_intervals)

    def _get_eval_model(self) -> nn.Module:
        """Return an eval-mode, CPU-resident copy of the currently-training model,
        detached from anything the training thread goes on to mutate. Call this
        only from the training thread, then hand the copy to other threads."""
        model_copy = deepcopy(self._get_live_model()).cpu()
        model_copy.eval()
        return model_copy

    def _write_network_template(self, dest: Path) -> None:
        """Pickle the current model to `dest` as the run's architecture template,
        which loaders pair with each checkpoint's ``weights.pt``."""
        model = self._get_live_model()
        CheckpointUtils.atomic_write(dest, lambda temp_path: torch.save(model, temp_path))

    def _log_network_summary(self) -> None:
        model = self._get_live_model()
        if model is None:
            return
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bar = "=" * 70
        sep = "-" * 70
        logger.info(
            f"\n{bar}\n\nNetwork architecture\n{sep}\n{model}\n{sep}\n"
            f"Total parameters:     {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n{bar}\n"
        )
