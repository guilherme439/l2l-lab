import queue
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from torch import nn

from l2l_lab._utils.checkpoint import CheckpointUtils
from l2l_lab._utils.common import CommonUtils
from l2l_lab._utils.exception_handler import ExceptionHandler
from l2l_lab._utils.graphs import GraphsUtils
from l2l_lab._utils.memory import MemorySampler
from l2l_lab._utils.wandb import WandbUtils
from l2l_lab.backends import get_backend
from l2l_lab.backends.backend_base import StepResult
from l2l_lab.configs.training.training_config import TrainingConfig
from l2l_lab.reporting import Reporter
from l2l_lab.testing.tester import GameResults
from l2l_lab.training.eval_worker import EvalRequest, EvalResult, EvalWorker
from l2l_lab.training.evaluator import Evaluator
from l2l_lab.training.metrics_store import MetricsStore, MetricsView
import logging

logger = logging.getLogger("l2l_lab")

MODELS_DIR = Path("models")

_DESTRUCTIVE_ABORT_WAIT_SECONDS = 30
_PLOT_MAX_POINTS = 2000


class Trainer:

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = TrainingConfig.from_yaml(config_path)
        self.backend = get_backend(self.config.backend.name)()
        reports_to_capture = (
            self.config.reporting.sample_games_per_eval if self.config.reporting.enabled else 0
        )
        self.evaluator = Evaluator(
            self.config.evaluation, self.backend, self.config.env, self.config.network,
            reports_to_capture=reports_to_capture,
        )
        self._metrics_store: Optional[MetricsStore] = None
        self.current_model_dir: Optional[Path] = None
        self.reporter: Optional[Reporter] = None
        self.eval_worker: Optional[EvalWorker] = None
        self._completed = False
        self._memory_sampler: Optional[MemorySampler] = None
        self._wandb_enabled: bool = False
        self._is_rewind: bool = False

    def train(self) -> None:
        cfg = self.config
        backend_cfg = cfg.backend
        algo_cfg = backend_cfg.algorithm
        total_iterations = algo_cfg.total_iterations

        logger.info("\n" * 3)
        logger.info("=" * 70)
        logger.info(f"\nTraining {cfg.env.name.upper()} with {algo_cfg.name.upper()} (backend: {backend_cfg.name})\n")
        logger.info(f"  Name: {cfg.name}")
        logger.info("")
        logger.info("=" * 70)

        if cfg.common.checkpoint_interval <= 0:
            yellow_color_tags = "\033[33m", "\033[0m"
            start, end = yellow_color_tags
            logger.warning(
                f"{start}\n"
                "⚠ Checkpointing is disabled - "
                "'checkpoint_interval' was set to 0 or not set at all ⚠\n"
                f"{end}"
            )

        if cfg.common.plot_memory:
            self._memory_sampler = MemorySampler()

        model_dir = MODELS_DIR / cfg.name

        self.backend.init()

        if backend_cfg.continue_training:
            self._setup_model_dir(model_dir)
            loaded_iteration = self.backend.restore_run(cfg, self.current_model_dir)
            self._is_rewind = CheckpointUtils.is_rewind(self.current_model_dir, loaded_iteration)
            if self._is_rewind:
                self._clear_rewinded_iterations(loaded_iteration, _DESTRUCTIVE_ABORT_WAIT_SECONDS)
            resume_from = loaded_iteration
            starting_iteration = loaded_iteration + 1
        else:
            if model_dir.exists():
                self._clear_existing_model_dir(model_dir, _DESTRUCTIVE_ABORT_WAIT_SECONDS)

            self._setup_model_dir(model_dir)
            resume_from = None
            starting_iteration = 0
            self.backend.new_run(cfg, self.current_model_dir)

        self.backend.configure_checkpointing(
            self.current_model_dir,
            cfg.common.checkpoint_interval,
            self.evaluator.training_eval_intervals(),
            cfg.reporting.interval if cfg.reporting.enabled else 0,
        )

        self._setup_reporter(cfg, starting_iteration)
        self._setup_metrics_store(resume=backend_cfg.continue_training, truncate_to=resume_from)

        self._wandb_enabled = WandbUtils.init(
            run_name=cfg.name,
            training_config=cfg,
        )

        remaining_iterations = total_iterations - starting_iteration
        if remaining_iterations <= 0:
            logger.info(f"\nAlready completed {starting_iteration} iterations (target: {total_iterations}). Nothing to do.")
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"\n\nStarting training for {remaining_iterations} iterations (from {starting_iteration} to {total_iterations})...")
        logger.info("Press Ctrl+C to stop early.")
        logger.info("-" * 70)
        self._starting_iteration = starting_iteration
        self._last_completed_iteration = starting_iteration - 1
        self._completed = False
        self._evals_in_flight = 0
        self._deferred_snapshots: list[tuple[int, Optional[nn.Module]]] = []
        self._previous_checkpoint: Optional[Path] = None
        self.eval_worker = EvalWorker(self.evaluator, self.backend)
        self.backend.start_training()

        with ExceptionHandler(self._graceful_shutdown):
            self._run_training_loop()

    def load_metrics(self) -> MetricsView:
        """Read the full persisted metrics history back into memory."""
        if self._metrics_store is None:
            raise RuntimeError("Metrics store is not initialized.")
        return self._metrics_store.load_view()

    def plot_progress(self, plot_memory: bool = True) -> None:
        if self._metrics_store is None:
            raise RuntimeError("Metrics store is not initialized.")
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")

        view = self._metrics_store.load_view(max_points=_PLOT_MAX_POINTS)
        if not view.scalars.get("iteration"):
            logger.info("No metrics to plot!")
            return

        graphs_dir = self.current_model_dir / "graphs"
        GraphsUtils.plot_metrics(graphs_dir, view, self.config.common.eval_graph_split, plot_memory=plot_memory)
        logger.info(f"\n\n📊 Graphs saved to: {graphs_dir}\n\n")

    def _run_training_loop(self) -> None:
        while True:
            try:
                step_result = self.backend.step_queue.get(timeout=1)
            except queue.Empty:
                self._drain_eval_results()
                self._flush_deferred_snapshots()
                continue

            if step_result is None:
                self._completed = True
                break

            self._process_step(step_result)
            self._drain_eval_results()
            self._flush_deferred_snapshots()

    def _process_step(self, step_result: StepResult) -> None:
        """Record a finished step's metrics and, if evals or a reporting
        snapshot are due, hand them off to the eval/reporter workers. Never
        blocks on game play, file I/O, or probe inference."""
        cfg = self.config
        current_iteration = step_result.iteration
        iterations_completed = current_iteration + 1
        self._last_completed_iteration = current_iteration
        step_metrics = step_result.metrics
        logger.info(f"Processing step {current_iteration} result")

        self._collect_weight_metrics(step_metrics)
        if self._memory_sampler is not None:
            self._collect_memory_metrics(step_metrics)

        self._metrics_store.record_step(current_iteration, step_metrics)

        if self.reporter is not None:
            self.reporter.on_step(current_iteration, step_metrics)

        if self._wandb_enabled:
            WandbUtils.log(step_metrics, current_iteration)

        needs_eval = (
            self.evaluator.training_evals_due(iterations_completed) or step_result.checkpoint_path is not None
        )
        if needs_eval and self.eval_worker is not None:
            self.eval_worker.enqueue(EvalRequest(
                iteration=current_iteration,
                eval_model=step_result.eval_model,
                checkpoint_path=step_result.checkpoint_path,
                previous_checkpoint=self._previous_checkpoint,
            ))
            self._evals_in_flight += 1

        if step_result.checkpoint_path is not None:
            self._previous_checkpoint = step_result.checkpoint_path

        if self.reporter is not None and self.reporter.snapshot_due(iterations_completed):
            self._deferred_snapshots.append((current_iteration, step_result.eval_model))

        if CommonUtils.check_interval(iterations_completed, cfg.common.plot_interval):
            self.plot_progress()

    def _process_remaining_steps(self) -> None:
        """Process every step still queued by the backend, so a stop mid-run
        records metrics, evals, and checkpoints for steps that finished
        training but hadn't been picked up yet."""
        while True:
            try:
                step_result = self.backend.step_queue.get_nowait()
            except queue.Empty:
                break
            if step_result is None:
                continue
            self._process_step(step_result)

    def _drain_eval_results(self) -> None:
        if self.eval_worker is None:
            return
        for result in self.eval_worker.drain_results():
            self._merge_eval_result(result)
            self._evals_in_flight -= 1

    def _merge_eval_result(self, result: EvalResult) -> None:
        """Persist a completed eval into the metrics store, then log, report,
        and run the checkpoint hook it unblocks."""
        for label, game_result in result.results.items():
            if game_result is None:
                continue
            for position, side in (("as_p0", game_result.as_p0), ("as_p1", game_result.as_p1)):
                if side is None:
                    continue
                self._metrics_store.record_eval(
                    result.iteration, label, position, side.wins, side.losses, side.draws
                )
                if self.reporter is not None:
                    for report in side.reports:
                        self.reporter.add_game_report(result.iteration, label, position, report)

        eval_str = self._format_eval_results(result.results)
        if eval_str:
            logger.info("\n")
            logger.info("  ┌─ Eval Results ───────────────────────────────")
            logger.info(eval_str)
            logger.info("  └──────────────────────────────────────────────")
            logger.info("")

        if self._wandb_enabled:
            WandbUtils.log_evaluations(self._build_eval_wandb_metrics(result.results), result.iteration)

        if result.checkpoint_path is not None:
            self.backend.on_checkpoint_saved(self.current_model_dir, result.iteration)

    def _flush_deferred_snapshots(self) -> None:
        """Emit reporting snapshots that were waiting on in-flight evals, now
        that every eval up to the latest deferred iteration has landed."""
        if self._evals_in_flight > 0 or not self._deferred_snapshots or self.reporter is None:
            return
        for iteration, model in self._deferred_snapshots:
            view = self._metrics_store.load_view(up_to=iteration, tail=self.reporter.sparkline_window)
            self.reporter.emit_snapshot(iteration, view, model)
        self._deferred_snapshots = []

    def _graceful_shutdown(self) -> None:
        cfg = self.config
        total_iterations = cfg.backend.algorithm.total_iterations
        algo_name = cfg.backend.algorithm.name.upper()

        self.backend.request_stop()
        self.backend.wait_for_training(cfg.backend.graceful_shutdown_period)
        self._process_remaining_steps()
        last_completed_iteration = self._last_completed_iteration

        try:
            logger.info("")
            logger.info("-" * 70)
            if self._completed:
                logger.info(f"✓ {algo_name} Training completed!")
            else:
                logger.info(f"Training stopped early at iteration {last_completed_iteration}/{total_iterations}.")

            if self.eval_worker is not None:
                logger.info("\nWaiting for pending evaluations to finish...\n")
                self.eval_worker.wait_for_idle()
                self._drain_eval_results()
                self._flush_deferred_snapshots()

            self._save_final_checkpoint(last_completed_iteration)
            self.plot_progress(plot_memory=False)
        except Exception:
            logger.exception("Error during shutdown while saving the final checkpoints and plots")

        if self.eval_worker is not None:
            self.eval_worker.stop()
        self.backend.shutdown()
        if self.reporter is not None:
            self.reporter.on_shutdown()
        if self._wandb_enabled:
            WandbUtils.finish()
        if self._metrics_store is not None:
            self._metrics_store.close()

    def _save_final_checkpoint(self, iteration: int) -> None:
        if iteration < self._starting_iteration:
            return
        self.backend.save_final_checkpoint(iteration)

    def _setup_model_dir(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        (model_dir / "graphs" / "evaluations").mkdir(exist_ok=True)
        self.current_model_dir = model_dir

    def _setup_reporter(self, cfg: TrainingConfig, starting_iteration: int) -> None:
        if self.current_model_dir is None:
            raise RuntimeError("Model directory must be set before wiring the reporter.")
        reports_dir = self.current_model_dir / "reports"
        self.reporter = Reporter(
            cfg=cfg.reporting,
            run_name=cfg.name,
            backend_name=cfg.backend.name,
            env_config=cfg.env,
            config_path=self.config_path,
            reports_dir=reports_dir,
            resume=cfg.backend.continue_training,
            csv_keys=self.backend.get_reporter_csv_keys(),
        )
        self.reporter.on_setup(starting_iteration=starting_iteration)

    def _setup_metrics_store(self, resume: bool, truncate_to: Optional[int]) -> None:
        if self.current_model_dir is None:
            raise RuntimeError("Model directory must be set before wiring the metrics store.")
        self._metrics_store = MetricsStore(
            metrics_dir=self.current_model_dir / "metrics",
            label_types=self.evaluator.label_to_type_map(),
            resume=resume,
            truncate_to=truncate_to,
        )

    def _collect_memory_metrics(self, step_metrics: dict[str, Any]) -> None:
        sample = self._memory_sampler.sample()
        step_metrics["memory"] = {
            "main_pss_mb": sample.main_pss_mb,
            "workers_pss_mb": sample.workers_pss_mb,
            "total_pss_mb": sample.total_pss_mb,
        }

    def _collect_weight_metrics(self, step_metrics: dict[str, Any]) -> None:
        parameters = self.backend.get_weight_parameters()
        if parameters is None:
            return
        max_abs = 0.0
        min_abs = float("inf")
        abs_sum = 0.0
        count = 0
        for parameter in parameters:
            abs_values = parameter.data.abs()
            numel = abs_values.numel()
            if numel == 0:
                continue
            max_abs = max(max_abs, abs_values.max().item())
            min_abs = min(min_abs, abs_values.min().item())
            abs_sum += abs_values.sum().item()
            count += numel
        if count == 0:
            step_metrics["weight_max"] = 0.0
            step_metrics["weight_min"] = 0.0
            step_metrics["weight_avg"] = 0.0
            return
        step_metrics["weight_max"] = max_abs
        step_metrics["weight_min"] = min_abs
        step_metrics["weight_avg"] = abs_sum / count

    def _format_eval_results(self, results: dict[str, Optional[GameResults]]) -> str:
        formatted = [
            self._format_eval_line(label, result)
            for label, result in results.items()
            if result is not None
        ]
        return "\n".join(formatted)

    def _build_eval_wandb_metrics(self, results: dict[str, Optional[GameResults]]) -> dict[str, Any]:
        """Build the nested ``evaluations/...`` payload logged to wandb, holding
        only the labels present in `results`."""
        label_to_type = self.evaluator.label_to_type_map()
        buckets: dict[str, dict[str, Any]] = {"training": {}, "checkpoint": {}}
        for label, result in results.items():
            if result is None:
                continue
            buckets[label_to_type[label]][label] = {
                position: {"wins": side.wins, "losses": side.losses, "draws": side.draws}
                for position, side in (("as_p0", result.as_p0), ("as_p1", result.as_p1))
                if side is not None
            }
        return {"evaluations": buckets}

    def _clear_existing_model_dir(self, model_dir: Path, wait_seconds: int) -> None:
        red_color_tags = "\033[31m", "\033[0m"
        start, end = red_color_tags
        logger.warning(
            f"{start}\nModel directory {model_dir} already exists.\n"
            "!! Directory will be cleared before starting a fresh training run !!\n"
            f" - Press Ctrl+C within {wait_seconds}s to abort.\n\n{end}"
        )
        time.sleep(wait_seconds)
        shutil.rmtree(model_dir)

    def _clear_rewinded_iterations(self, loaded_iteration: int, wait_seconds: int) -> None:
        stale_iters = CheckpointUtils.list_checkpoint_iterations_past(self.current_model_dir, loaded_iteration)
        reports_dir = self.current_model_dir / "reports"

        red_color_tags = "\033[31m", "\033[0m"
        start, end = red_color_tags
        logger.warning(
            f"{start}\nRewinding to iteration {loaded_iteration} "
            f"while later iterations exist on disk: {stale_iters}.\n"
            f"!! Checkpoints past {loaded_iteration} and any report data will be removed !!\n"
            f" - Press Ctrl+C within {wait_seconds}s to abort.\n\n{end}"
        )
        time.sleep(wait_seconds)

        self.backend.delete_checkpoints_past(self.current_model_dir, loaded_iteration)
        Reporter.clear_artifacts_past(reports_dir, loaded_iteration)

    @staticmethod
    def _format_eval_line(label: str, result: GameResults) -> str:
        lines = [
            f"  │ {label}: {result.wins}W/{result.losses}L/{result.draws}D"
            f" - {result.win_rate:.0%}/{result.loss_rate:.0%}/{result.draw_rate:.0%}"
            f" (avg: {result.avg_moves:.1f})"
        ]
        if result.as_p0 and result.as_p1:
            p0, p1 = result.as_p0, result.as_p1
            lines.append(
                f"  │   P0: {p0.wins}W/{p0.losses}L/{p0.draws}D"
                f" ({p0.win_rate:.0%}) avg:{p0.avg_moves:.1f}"
                f" | P1: {p1.wins}W/{p1.losses}L/{p1.draws}D"
                f" ({p1.win_rate:.0%}) avg:{p1.avg_moves:.1f}"
            )
        return "\n".join(lines)
