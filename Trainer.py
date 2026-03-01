from __future__ import annotations

import queue
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import shutil

import graphs
from backends import get_backend
from configs.definition.training.TrainingConfig import TrainingConfig
from Tester import Tester

MODELS_DIR = Path("models")


class Trainer:

    def __init__(self, config_path: Union[str, Path]):
        self.config = TrainingConfig.from_yaml(config_path)
        self.backend = get_backend(self.config.backend)()
        self.metrics: Dict[str, List] = {}
        self.current_model_dir: Optional[Path] = None
        self._early_stop_requested = False

    def _handle_sigint(self, signum, frame):
        if self._early_stop_requested:
            print("\nForce stopping...")
            sys.exit(1)
        self._early_stop_requested = True
        print("\n\nStopping after current step completes... (Ctrl+C again to force quit)\n")

    def _setup_model_dir(self) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        model_dir = MODELS_DIR / f"{cfg.name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "graphs").mkdir(exist_ok=True)
        self.current_model_dir = model_dir
        return model_dir

    def _init_metrics(self) -> None:
        self.metrics = {
            "iteration": [],
            "wins_vs_random": [],
            "losses_vs_random": [],
            "draws_vs_random": [],
            "weight_max": [],
            "weight_min": [],
            "weight_avg": [],
        }
        if self.config.eval_vs_previous:
            self.metrics["wins_vs_previous"] = []
            self.metrics["losses_vs_previous"] = []
            self.metrics["draws_vs_previous"] = []

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

    def _record_eval(self, results, prefix: str) -> str:
        w_key, l_key, d_key = f"wins_{prefix}", f"losses_{prefix}", f"draws_{prefix}"

        if w_key in self.metrics:
            self.metrics[w_key].append(results.wins if results else None)
            self.metrics[l_key].append(results.losses if results else None)
            self.metrics[d_key].append(results.draws if results else None)

        if results:
            line = (f"\n\n{' ' * 32}"
                    f" | {prefix}: {results.wins}W/{results.losses}L/{results.draws}D"
                    f" - {results.win_rate:.0%}/{results.loss_rate:.0%}/{results.draw_rate:.0%}"
                    f" (avg: {results.avg_moves:.1f})")
            if results.as_p0 and results.as_p1:
                p0, p1 = results.as_p0, results.as_p1
                line += (f"\n{' ' * 32}"
                         f"   P0: {p0.wins}W/{p0.losses}L/{p0.draws}D"
                         f" ({p0.win_rate:.0%}) avg:{p0.avg_moves:.1f}"
                         f" | P1: {p1.wins}W/{p1.losses}L/{p1.draws}D"
                         f" ({p1.win_rate:.0%}) avg:{p1.avg_moves:.1f}")
            return line
        return ""

    def train(self) -> None:
        cfg = self.config
        algo_cfg = cfg.algorithm

        print("\n" * 3)
        print("=" * 70)
        print(f"\nTraining {cfg.env.name.upper()} with {algo_cfg.name.upper()} (backend: {cfg.backend})\n")
        print(f"  Name: {cfg.name}")
        print()
        print("=" * 70)

        self._setup_model_dir()

        if cfg.continue_training:
            start_iteration, cp_data = self.backend.restore(
                cfg, self.current_model_dir, self.current_model_dir
            )
            if cp_data and cp_data.metrics:
                self.metrics = cp_data.metrics
                print(f"✓ Loaded {len(self.metrics.get('iteration', []))} iterations of metrics from checkpoint")
            else:
                self._init_metrics()
        else:
            if self.current_model_dir.exists():
                # clear dir if it exists to avoid confusion
                shutil.rmtree(self.current_model_dir)
                self._setup_model_dir()
            start_iteration = 0
            self.backend.setup(cfg, self.current_model_dir)
            self._init_metrics()

        previous_checkpoint: Optional[Path] = None
        metrics_initialized = len(self.metrics.get("iteration", [])) > 0

        remaining_iterations = algo_cfg.iterations - start_iteration
        if remaining_iterations <= 0:
            print(f"\nAlready completed {start_iteration} iterations (target: {algo_cfg.iterations}). Nothing to do.")
            return

        print()
        print("=" * 70)
        print(f"\n\nStarting training for {remaining_iterations} iterations (from {start_iteration} to {algo_cfg.iterations})...")
        print("Press Ctrl+C to stop early.")
        print("-" * 70)

        self.backend.start_training(start_iteration, algo_cfg.iterations)

        latest_checkpoint_data = None
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
                if step_result.checkpoint_data is not None:
                    latest_checkpoint_data = step_result.checkpoint_data

                # Remove internal keys before storing
                rllib_result = metrics.pop("_rllib_result", None)

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

                eval_str = ""

                results_random = None
                if i % cfg.eval_interval == 0:
                    agent = self.backend.create_eval_agent()
                    results_random = Tester.evaluate_agent_vs_random(agent, cfg.env, num_games=cfg.eval_games)
                    if hasattr(self.backend, '_set_module_training'):
                        self.backend._set_module_training()
                eval_str += self._record_eval(results_random, "vs_random")

                results_prev = None
                if i % cfg.checkpoint_interval == 0:
                    if cfg.eval_vs_previous and previous_checkpoint is not None:
                        agent = self.backend.create_eval_agent()
                        opponent = self.backend.create_agent_from_checkpoint(previous_checkpoint)
                        results_prev = Tester.evaluate_agent_vs_agent(
                            agent, opponent, cfg.env, num_games=cfg.eval_games
                        )
                        if hasattr(self.backend, '_set_module_training'):
                            self.backend._set_module_training()
                    checkpoint_dir = self.save_checkpoint(i, latest_checkpoint_data)
                    previous_checkpoint = checkpoint_dir

                    if hasattr(self.backend, 'update_opponent_policies'):
                        self.backend.update_opponent_policies(self.current_model_dir, i)
                eval_str += self._record_eval(results_prev, "vs_previous")

                ep_len = metrics.get('episode_len_mean', 0) or 0
                print(f"{i:8d}/{algo_cfg.iterations} | EpLen: {ep_len:6.1f}{eval_str}")

                if i % cfg.plot_interval == 0:
                    self.plot_progress()

                if i % cfg.info_interval == 0 and rllib_result is not None:
                    if hasattr(self.backend, 'print_training_info'):
                        self.backend.print_training_info(rllib_result)

                if self._early_stop_requested:
                    break
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        if i < algo_cfg.iterations or self._early_stop_requested:
            print("-" * 70)
            print(f"Training stopped early at iteration {i}/{algo_cfg.iterations}.")
            if latest_checkpoint_data is not None:
                self.save_checkpoint(i, latest_checkpoint_data)
            self.plot_progress()
            self.backend.shutdown()
            return

        print("-" * 70)
        print(f"✓ {algo_cfg.name.upper()} Training completed!")

        self.save_checkpoint(algo_cfg.iterations, latest_checkpoint_data)
        self.backend.shutdown()
        self.plot_progress()

    def save_checkpoint(self, iteration: int, checkpoint_data: Dict[str, Any]) -> Path:
        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")

        checkpoint_dir = self.current_model_dir / "checkpoints" / str(iteration)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.backend.save_checkpoint(checkpoint_dir, iteration, self.metrics, checkpoint_data)

        print(f"\n  [Checkpoint saved: iter {iteration}]\n")
        return checkpoint_dir

    def get_latest_checkpoint(self, model_dir: Path) -> Optional[Path]:
        from checkpoint_utils import get_latest_checkpoint_dir
        return get_latest_checkpoint_dir(model_dir)

    def plot_progress(self) -> None:
        if not self.metrics.get("iteration"):
            print("No metrics to plot!")
            return

        if self.current_model_dir is None:
            raise RuntimeError("No model directory.")

        graphs_dir = self.current_model_dir / "graphs"
        graphs.plot_metrics(graphs_dir, self.metrics, self.config.eval_graph_split)
        print(f"\n📊 Graphs saved to: {graphs_dir}\n")
