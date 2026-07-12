import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from torch import nn

    from l2l_lab.backends.backend_base import AlgorithmBackend
    from l2l_lab.testing.tester import GameResults
    from l2l_lab.training.evaluator import Evaluator


@dataclass
class EvalRequest:
    iteration: int
    eval_model: Optional[nn.Module]
    checkpoint_path: Optional[Path]
    previous_checkpoint: Optional[Path]


@dataclass
class EvalResult:
    iteration: int
    results: dict[str, Optional[GameResults]]
    checkpoint_path: Optional[Path]


class EvalWorker:
    """Runs evaluations on a dedicated daemon thread so the caller never blocks
    on game play. Requests are processed one at a time, in the order enqueued,
    so results are completed in iteration order.
    """

    def __init__(self, evaluator: Evaluator, backend: AlgorithmBackend) -> None:
        self._evaluator = evaluator
        self._backend = backend
        self._requests: Queue[Optional[EvalRequest]] = Queue()
        self._results: Queue[EvalResult] = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, request: EvalRequest) -> None:
        self._requests.put(request)

    def drain_results(self) -> list[EvalResult]:
        """Return every `EvalResult` completed so far, without blocking."""
        drained: list[EvalResult] = []
        while True:
            try:
                drained.append(self._results.get_nowait())
            except queue.Empty:
                break
        return drained

    def wait_for_idle(self) -> None:
        """Block until every enqueued request has been processed."""
        self._requests.join()

    def stop(self) -> None:
        self._requests.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            try:
                if request is None:
                    return
                self._results.put(self._safe_process(request))
            finally:
                self._requests.task_done()

    def _safe_process(self, request: EvalRequest) -> EvalResult:
        """
        Run a request's evals, converting any failure into an empty result.

        A failed eval must still yield exactly one `EvalResult` (so the caller's
        in-flight accounting and `wait_for_idle` stay correct) and must not kill the worker thread. 
        """
        try:
            return self._process(request)
        except Exception:
            logger.exception(
                f"Evaluation for iteration {request.iteration} failed; skipping its results."
            )
            return EvalResult(iteration=request.iteration, results={}, checkpoint_path=request.checkpoint_path)

    def _process(self, request: EvalRequest) -> EvalResult:
        if request.checkpoint_path is not None:
            self._backend.wait_for_pending_checkpoints()

        results = self._evaluator.run_training_evals(request.iteration, request.eval_model)
        if request.checkpoint_path is not None:
            checkpoint_results = self._evaluator.run_checkpoint_evals(
                request.previous_checkpoint, iteration=request.iteration, eval_model=request.eval_model
            )
            results = {**results, **checkpoint_results}

        return EvalResult(iteration=request.iteration, results=results, checkpoint_path=request.checkpoint_path)
