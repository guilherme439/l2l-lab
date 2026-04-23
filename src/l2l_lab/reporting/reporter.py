from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from l2l_lab.configs.common.EnvConfig import EnvConfig
from l2l_lab.configs.training.ReportingConfig import ReportingConfig
from l2l_lab.utils.common import check_interval

from .csv_writer import MetricsCSVWriter
from .markdown import StampedReport, render_report
from .probe_runner import run_probe_states
from .probe_states import get_probe_states
from .types import GameReport

logger = logging.getLogger("alphazoo")


class Reporter:
    """Orchestrates per-iteration CSV writes and periodic Markdown snapshots.

    When ``cfg.enabled`` is False all methods become no-ops and no files are
    touched. When enabled, writes live under ``reports_dir`` (typically
    ``models/{run_name}/reports/``): ``training.csv``, ``config.yaml``
    (archived verbatim from the source YAML), and ``report_{iter:06d}.md``
    snapshots every ``cfg.interval`` iterations.
    """

    def __init__(
        self,
        cfg: ReportingConfig,
        run_name: str,
        backend_name: str,
        env_config: EnvConfig,
        config_path: Path,
        reports_dir: Path,
        resume: bool,
    ) -> None:
        self.cfg = cfg
        self._run_name = run_name
        self._backend_name = backend_name
        self._env_config = env_config
        self._config_path = config_path
        self._reports_dir = reports_dir
        self._resume = resume

        self._pending_reports: list[StampedReport] = []
        self._csv_writer: Optional[MetricsCSVWriter] = None
        self._metrics_getter: Optional[Callable[[], dict[str, Any]]] = None
        self._model_getter: Optional[Callable[[], Any]] = None
        self._start_iteration: int = 0

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def add_game_report(
        self,
        iteration: int,
        eval_label: str,
        as_position: str,
        report: GameReport,
    ) -> None:
        """Buffer a captured game, stamped with the caller's eval context.

        Drained into the next Markdown snapshot; harmless if reporting is off.
        """
        if not self.cfg.enabled:
            return
        self._pending_reports.append(StampedReport(iteration, eval_label, as_position, report))

    def attach_sources(
        self,
        metrics_getter: Callable[[], dict[str, Any]],
        model_getter: Callable[[], Any],
    ) -> None:
        """Wire in lazy accessors for the full metrics dict and the eval model.

        The Reporter only invokes these at snapshot time, so the Trainer never
        has to hand over the full metrics dict on every step.
        """
        self._metrics_getter = metrics_getter
        self._model_getter = model_getter

    def on_setup(self, start_iteration: int = 0) -> None:
        if not self.cfg.enabled:
            return

        self._start_iteration = start_iteration
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._handle_config_archive()

        csv_path = self._reports_dir / "training.csv"
        self._csv_writer = MetricsCSVWriter(csv_path, resume=self._resume)
        logger.info("Reporter enabled: writing to %s", self._reports_dir)

    def on_step(self, iteration: int, step_metrics: dict[str, Any]) -> None:
        if not self.cfg.enabled or self._csv_writer is None:
            return

        self._csv_writer.append(iteration, step_metrics)

        if check_interval(iteration, self.cfg.interval):
            self._emit_snapshot(iteration)

    def on_shutdown(self) -> None:
        if self._csv_writer is not None:
            self._csv_writer.close()
            self._csv_writer = None

    def _emit_snapshot(self, iteration: int) -> None:
        metrics = self._metrics_getter() if self._metrics_getter else {}
        model = self._model_getter() if self._model_getter else None

        probe_states = get_probe_states(self._env_config.name) if self._env_config.name else []
        probe_results = run_probe_states(model, self._env_config, probe_states)

        reports = self._drain_reports()

        window = max(10, self.cfg.interval // 10)
        md = render_report(
            run_name=self._run_name,
            iteration=iteration,
            backend_name=self._backend_name,
            env_name=self._env_config.name or "unknown",
            metrics=metrics,
            probe_results=probe_results,
            reports=reports,
            sparkline_window=window,
        )

        out_path = self._reports_dir / f"report_{iteration:06d}.md"
        out_path.write_text(md, encoding="utf-8")
        logger.info("Reporter snapshot written: %s", out_path)

    def _drain_reports(self) -> list[StampedReport]:
        drained = self._pending_reports
        self._pending_reports = []
        return drained

    def _handle_config_archive(self) -> None:
        dest = self._reports_dir / "config.yaml"
        if not self._resume or not dest.exists():
            shutil.copyfile(self._config_path, dest)
            return

        archived_latest = self._latest_archived_config()
        try:
            current_bytes = self._config_path.read_bytes()
            archived_bytes = archived_latest.read_bytes()
        except OSError as e:
            logger.warning("Reporter: could not read config for hash diff: %s", e)
            return

        if Reporter._canonical_hash(current_bytes) == Reporter._canonical_hash(archived_bytes):
            return

        archive_name = f"config_{self._start_iteration:06d}.yaml"
        archive_path = self._reports_dir / archive_name
        archive_path.write_bytes(current_bytes)
        logger.info("Reporter: config change detected, archived to %s", archive_path)

    def _latest_archived_config(self) -> Path:
        """Return the most recent config snapshot to compare against."""
        iter_archives = []
        pattern = re.compile(r"^config_(\d{6})\.yaml$")
        for path in self._reports_dir.iterdir():
            m = pattern.match(path.name)
            if m:
                iter_archives.append((int(m.group(1)), path))
        if iter_archives:
            iter_archives.sort()
            return iter_archives[-1][1]
        return self._reports_dir / "config.yaml"

    @staticmethod
    def _canonical_hash(yaml_bytes: bytes) -> str:
        """Hash the structural content of a YAML blob, ignoring whitespace/ordering."""
        try:
            data = yaml.safe_load(yaml_bytes)
        except yaml.YAMLError:
            return hashlib.sha256(yaml_bytes).hexdigest()
        canonical = yaml.safe_dump(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
