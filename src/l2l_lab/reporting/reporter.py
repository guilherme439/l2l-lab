import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

import yaml

from l2l_lab._utils.common import CommonUtils
from l2l_lab._utils.csv_writer import MetricsCSVWriter
from l2l_lab.configs.common.env_config import EnvConfig
from l2l_lab.configs.training.reporting_config import ReportingConfig

from .markdown import StampedReport, render_report
from .probe_runner import run_probe_states
from .probe_states import get_probe_states
from .types import GameReport
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from torch import nn

    from l2l_lab.training.metrics_store import MetricsView

_REPORT_SNAPSHOT_PATTERN = re.compile(r"^report_(\d{6})\.md$")
_ARCHIVED_CONFIG_PATTERN = re.compile(r"^config_(\d{6})\.yaml$")


@dataclass
class _CsvRowRequest:
    iteration: int
    row: dict[str, Any]


@dataclass
class _SnapshotRequest:
    iteration: int
    view: MetricsView
    model: Optional[nn.Module]
    reports: list[StampedReport]


_ReporterRequest = _CsvRowRequest | _SnapshotRequest


class Reporter:
    """Buffers per-iteration metrics and periodic diagnostic snapshots, then
    writes them from a dedicated worker thread so callers never block on
    file I/O or probe-state inference.

    When ``cfg.enabled`` is False every method is a no-op and no files are
    touched. When enabled, writes land under ``reports_dir`` (typically
    ``models/{run_name}/reports/``): ``training.csv``, ``config.yaml``
    (archived verbatim from the source YAML), and ``report_{iter:06d}.md``
    snapshots for iterations the caller marks via `snapshot_due`.
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
        csv_keys: list[str],
    ) -> None:
        self.cfg = cfg
        self._run_name = run_name
        self._backend_name = backend_name
        self._env_config = env_config
        self._config_path = config_path
        self._reports_dir = reports_dir
        self._resume = resume
        self._csv_keys = set(csv_keys)

        self._pending_reports: list[StampedReport] = []
        self._csv_writer: Optional[MetricsCSVWriter] = None
        self._starting_iteration: int = 0

        self._requests: Queue[Optional[_ReporterRequest]] = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    @classmethod
    def clear_artifacts_past(cls, reports_dir: Path, iteration: int) -> None:
        """Remove report snapshots, archived config snapshots, and CSV rows past `iteration`."""
        if not reports_dir.exists():
            return
        for path, _ in CommonUtils.find_paths_with_iteration_past(reports_dir, _REPORT_SNAPSHOT_PATTERN, iteration):
            path.unlink()
        for path, _ in CommonUtils.find_paths_with_iteration_past(reports_dir, _ARCHIVED_CONFIG_PATTERN, iteration):
            path.unlink()
        MetricsCSVWriter.truncate_to_iteration(reports_dir / "training.csv", iteration)

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

    def snapshot_due(self, iterations_completed: int) -> bool:
        """True when `iterations_completed` lands on a reporting interval, so
        the caller knows to capture a model snapshot and call `emit_snapshot`."""
        return self.cfg.enabled and CommonUtils.check_interval(iterations_completed, self.cfg.interval)

    @property
    def sparkline_window(self) -> int:
        """Number of trailing points a snapshot renders per scalar series."""
        return max(10, self.cfg.interval // 10)

    def on_setup(self, starting_iteration: int = 0) -> None:
        if not self.cfg.enabled:
            return

        self._starting_iteration = starting_iteration
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._handle_config_archive()

        csv_path = self._reports_dir / "training.csv"
        self._csv_writer = MetricsCSVWriter(csv_path, resume=self._resume)
        logger.info(f"Reporter enabled: writing to {self._reports_dir}")

    def on_step(self, iteration: int, step_metrics: dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        csv_row = {k: v for k, v in step_metrics.items() if k in self._csv_keys}
        self._requests.put(_CsvRowRequest(iteration=iteration, row=csv_row))

    def emit_snapshot(self, iteration: int, view: MetricsView, model: Optional[nn.Module]) -> None:
        """Enqueue a Markdown snapshot for `iteration`, built from a metrics view
        the caller trimmed to that iteration and the model snapshot captured for
        this step. Call only when `snapshot_due` was true for this iteration.
        """
        if not self.cfg.enabled:
            return
        reports = self._drain_reports()
        self._requests.put(_SnapshotRequest(
            iteration=iteration, view=view, model=model, reports=reports,
        ))

    def on_shutdown(self) -> None:
        self._requests.put(None)
        self._thread.join()
        if self._csv_writer is not None:
            self._csv_writer.close()
            self._csv_writer = None

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            try:
                if request is None:
                    return
                if isinstance(request, _CsvRowRequest):
                    self._write_csv_row(request)
                else:
                    self._write_snapshot(request)
            finally:
                self._requests.task_done()

    def _write_csv_row(self, request: _CsvRowRequest) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.append(request.iteration, request.row)

    def _write_snapshot(self, request: _SnapshotRequest) -> None:
        probe_states = get_probe_states(self._env_config.name) if self._env_config.name else []
        probe_results = run_probe_states(request.model, self._env_config, probe_states)

        md = render_report(
            run_name=self._run_name,
            iteration=request.iteration,
            backend_name=self._backend_name,
            env_name=self._env_config.name or "unknown",
            scalars=request.view.scalars,
            evaluations=request.view.evaluations,
            probe_results=probe_results,
            reports=request.reports,
            sparkline_window=self.sparkline_window,
        )

        out_path = self._reports_dir / f"report_{request.iteration:06d}.md"
        out_path.write_text(md, encoding="utf-8")
        logger.info(f"\nReporter snapshot written: {out_path}")

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
            logger.warning(f"WARNING: Reporter: could not read config for hash diff: {e}")
            return

        if Reporter._canonical_hash(current_bytes) == Reporter._canonical_hash(archived_bytes):
            return

        archive_name = f"config_{self._starting_iteration:06d}.yaml"
        archive_path = self._reports_dir / archive_name
        archive_path.write_bytes(current_bytes)
        logger.info(f"Reporter: config change detected, archived to {archive_path}")

    def _latest_archived_config(self) -> Path:
        """Return the most recent config snapshot to compare against."""
        iter_archives: list[tuple[int, Path]] = []
        for path in self._reports_dir.iterdir():
            m = _ARCHIVED_CONFIG_PATTERN.match(path.name)
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
