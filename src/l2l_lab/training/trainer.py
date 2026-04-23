from __future__ import annotations

import logging
import queue
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from l2l_lab.backends import get_backend
from l2l_lab.backends.backend_base import StepResult
from l2l_lab.configs.training.TrainingConfig import TrainingConfig
from l2l_lab.reporting import Reporter
from l2l_lab.testing.tester import GameResults
from l2l_lab.training.evaluator import Evaluator
from l2l_lab.utils import graphs
from l2l_lab.utils.checkpoint import get_latest_checkpoint_dir
from l2l_lab.utils.common import check_interval

MODELS_DIR = Path("models")

logger = logging.getLogger("alphazoo")


class Trainer:

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = TrainingConfig.from_yaml(config_path)
        self.backend = get_backend(self.config.backend.name)()
        self.evaluator = Evaluator(self.config.evaluation, self.backend, self.config.env)
        self.metrics: Dict[str, Any] = {}
        self.current_model_dir: Optional[Path] = None
        self.reporter: Optional[Reporter] = None
        self._early_stop_requested = False

    def train(self) -> None:
        cfg = self.config
        backend_cfg = cfg.backend
        algo_cfg = backend_cfg.algorithm
        total_iterations = algo_cfg.total_iterations

        print("\n" * 3)
        print("=" * 70)
        print(f"\nTraining {cfg.env.name.upper()} with {algo_cfg.name.upper()} (backend: {backend_cfg.name})\n")
        print(f"  Name: {cfg.name}")
        print()
        print("=" * 70)

        if cfg.common.checkpoint_interval <= 0:
            yellow_color_tags = "\033[33m", "\033[0m"
            start, end = yellow_color_tags
            logger.warning(
                f"{start}\n"
                "⚠ Checkpointing is disabled - "
                "'checkpoint_interval' was set to 0 or not set at all ⚠\n"
                f"{end}"
            )

        self._init_metrics()

        model_dir = MODELS_DIR / cfg.name

        if backend_cfg.continue_training:
            self._setup_model_dir(model_dir)
            start_iteration, cp_data = self.backend.restore(
                cfg, self.current_model_dir, self.current_model_dir
            )
            if cp_data and cp_data.metrics:
                self._load_metrics(cp_data.metrics)
                print(f"✓ Loaded {len(self.metrics.get('iteration', []))} iterations of metrics from checkpoint")
        else:
            if model_dir.exists():
                red_color_tags = "\033[31m", "\033[0m"
                start, end = red_color_tags
                logger.info(
                    f"{start}\nModel directory {model_dir} already exists.\n"
                    "!! Directory will be cleared before starting a fresh training run !!\n"
                    f" - Press Ctrl+C within 20s to abort.\n\n{end}"
                )
                time.sleep(20)
                shutil.rmtree(model_dir)

            self._setup_model_dir(model_dir)
            start_iteration = 0
            self.backend.setup(cfg, self.current_model_dir)

        self._setup_reporter(cfg, start_iteration)

        previous_checkpoint: Optional[Path] = None
        metrics_initialized = len(self.metrics.get("iteration", [])) > 0

        remaining_iterations = total_iterations - start_iteration
        if remaining_iterations <= 0:
            print(f"\nAlready completed {start_iteration} iterations (target: {total_iterations}). Nothing to do.")
            return

        print()
        print("=" * 70)
        print(f"\n\nStarting training for {remaining_iterations} iterations (from {start_iteration} to {total_iterations})...")
        print("Press Ctrl+C to stop early.")
        print("-" * 70)

        self.backend.start_training(start_iteration, total_iterations)

        i = start_iteration

        self._early_stop_requested = False
        original_sigint = signal.signal(signal.SIGINT, self._handle_sigint)

        try:
            while True:
                try:
                    step_result = self.backend.step_queue.get(timeout=1)
                except queue.Empty:
                    if self._early_stop_requested:
                        break
                    continue

                if step_result is None:
                    break

                i = step_result.iteration
                metrics = step_result.metrics

                weight_params = self.backend.get_weight_parameters()
                if weight_params is not None:
                    weight_stats = Trainer._collect_weight_stats(weight_params)
                    metrics.update(weight_stats)

                if not metrics_initialized:
                    for key in metrics.keys():
                        self.metrics[key] = []
                    metrics_initialized = True

                self.metrics["iteration"].append(i)
                for key, value in metrics.items():
                    if key in self.metrics:
                        self.metrics[key].append(value)
                    else:
                        self.metrics[key] = [None] * (len(self.metrics["iteration"]) - 1) + [value]

                if self.reporter is not None:
                    self.reporter.on_step(i, metrics)

                eval_str = ""
                training_results = self.evaluator.run_training_evals(i)

                checkpoint_results: Dict[str, Optional[GameResults]] = {}
                if check_interval(i, cfg.common.checkpoint_interval):
                    checkpoint_results = self.evaluator.run_checkpoint_evals(previous_checkpoint, iteration=i)

                eval_str += self._record_evals({**training_results, **checkpoint_results})

                if check_interval(i, cfg.common.checkpoint_interval):
                    checkpoint_dir = self.save_checkpoint(i, self.backend.get_checkpoint_data())
                    previous_checkpoint = checkpoint_dir
                    self.backend.on_checkpoint_saved(self.current_model_dir, i)

                if eval_str:
                    print("\n")
                    print("  ┌─ Eval Results ──────────────────────────────")
                    print(eval_str)
                    print("  └──────────────────────────────────────────────")
                    print()

                if check_interval(i, cfg.common.plot_interval):
                    self.plot_progress()

                if self._early_stop_requested:
                    break

            self.backend.request_stop()
            self.backend.wait_for_training()

            if i < total_iterations or self._early_stop_requested:
                # on early stop, the trainer thread queue might be delayed, so we skip to the end
                last_step = self._skip_to_end_of_queue()
                if last_step is not None:
                    i = last_step.iteration
                print("-" * 70)
                print(f"Training stopped early at iteration {i}/{total_iterations}.")
                self.save_checkpoint(i, self.backend.get_checkpoint_data())
                self.plot_progress()
                self.backend.shutdown()
                return

            print("-" * 70)
            print(f"✓ {algo_cfg.name.upper()} Training completed!")

            self.save_checkpoint(total_iterations, self.backend.get_checkpoint_data())
            self.backend.shutdown()
            self.plot_progress()
        finally:
            if self.reporter is not None:
                self.reporter.on_shutdown()
            signal.signal(signal.SIGINT, original_sigint)

    def save_checkpoint(self, iteration: int, checkpoint_data: Dict[str, Any]) -> Path:
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")

        checkpoint_dir = self.current_model_dir / "checkpoints" / str(iteration)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.backend.save_checkpoint(checkpoint_dir, iteration, self.metrics, checkpoint_data)

        print(f"\n  [Checkpoint saved: iter {iteration}]\n")
        return checkpoint_dir

    def get_latest_checkpoint(self, model_dir: Path) -> Optional[Path]:
        return get_latest_checkpoint_dir(model_dir)

    def plot_progress(self) -> None:
        if not self.metrics.get("iteration"):
            print("No metrics to plot!")
            return

        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")

        graphs_dir = self.current_model_dir / "graphs"
        graphs.plot_metrics(graphs_dir, self.metrics, self.config.common.eval_graph_split)
        print(f"\n\n📊 Graphs saved to: {graphs_dir}\n\n")

    @staticmethod
    def _collect_weight_stats(parameters) -> Dict[str, float]:
        all_params = []
        for p in parameters:
            all_params.append(p.data.cpu().numpy().ravel())
        if not all_params:
            return {"weight_max": 0.0, "weight_min": 0.0, "weight_avg": 0.0}
        all_weights = np.concatenate(all_params)
        return {
            "weight_max": float(np.max(np.abs(all_weights))),
            "weight_min": float(np.min(np.abs(all_weights))),
            "weight_avg": float(np.mean(np.abs(all_weights))),
        }

    def _handle_sigint(self, signum, frame):
        if self._early_stop_requested:
            print("\nForce stopping...")
            sys.exit(1)
        self._early_stop_requested = True
        print("\n\nStopping after current step completes... (Ctrl+C again to force quit)\n")

    def _skip_to_end_of_queue(self) -> Optional[StepResult]:
        """Drain the backend step queue, returning the most recent StepResult."""
        last: Optional[StepResult] = None
        while True:
            try:
                item = self.backend.step_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            last = item
        return last

    def _setup_model_dir(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        self.current_model_dir = model_dir

    def _setup_reporter(self, cfg: TrainingConfig, start_iteration: int) -> None:
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
        )
        self.reporter.attach_sources(
            metrics_getter=lambda: self.metrics,
            model_getter=self.backend.get_eval_model,
        )
        self.reporter.on_setup(start_iteration=start_iteration)
        self.evaluator.reporter = self.reporter

    def _init_metrics(self) -> None:
        label_types = self.evaluator.label_types()
        evaluations: Dict[str, Dict[str, Any]] = {
            label: {
                "type": label_types[label],
                "as_p0": {"wins": [], "losses": [], "draws": []},
                "as_p1": {"wins": [], "losses": [], "draws": []},
            }
            for label in self.evaluator.labels()
        }
        self.metrics = {
            "iteration": [],
            "evaluations": evaluations,
            "weight_max": [],
            "weight_min": [],
            "weight_avg": [],
        }

    def _load_metrics(self, checkpoint_metrics: Dict[str, Any]) -> None:
        for key, values in checkpoint_metrics.items():
            self.metrics[key] = values
        # If the resumed config introduces new eval labels, back-fill them with Nones
        # so lengths stay aligned with "iteration".
        evaluations = self.metrics.setdefault("evaluations", {})
        pad_len = len(self.metrics.get("iteration", []))
        label_types = self.evaluator.label_types()
        for label in self.evaluator.labels():
            bucket = evaluations.setdefault(label, {})
            bucket["type"] = label_types[label]
            for position in ("as_p0", "as_p1"):
                sub = bucket.setdefault(position, {"wins": [], "losses": [], "draws": []})
                for series in ("wins", "losses", "draws"):
                    sub.setdefault(series, [])
                    if len(sub[series]) < pad_len:
                        sub[series] = sub[series] + [None] * (pad_len - len(sub[series]))

    def _record_evals(self, results: Dict[str, Optional[GameResults]]) -> str:
        evaluations = self.metrics["evaluations"]
        formatted: list[str] = []
        for label in self.evaluator.labels():
            result = results.get(label)
            bucket = evaluations[label]
            p0 = result.as_p0 if result is not None else None
            p1 = result.as_p1 if result is not None else None
            for position, half in (("as_p0", p0), ("as_p1", p1)):
                bucket[position]["wins"].append(half.wins if half else None)
                bucket[position]["losses"].append(half.losses if half else None)
                bucket[position]["draws"].append(half.draws if half else None)
            if result is not None:
                formatted.append(self._format_eval_line(label, result))
        return "\n".join(formatted)

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
