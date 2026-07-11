"""On-disk persistence for training metrics.

Dense per-iteration scalars stream to ``scalars.csv`` (one row per iteration)
and are never held in memory in full. Sparse evaluation results stream to
``evals.csv`` (one row per iteration/label/position) and are also kept in a
compact in-memory structure. Readers reconstruct a `MetricsView` on demand,
streaming and optionally downsampling the scalar file so peak memory stays
bounded regardless of run length.
"""

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from l2l_lab._utils.csv_writer import MetricsCSVWriter

_EVAL_POSITIONS = ("as_p0", "as_p1")


@dataclass
class EvalPoint:
    iteration: int
    wins: int
    losses: int
    draws: int


@dataclass
class EvalSeries:
    as_p0: list[EvalPoint] = field(default_factory=list)
    as_p1: list[EvalPoint] = field(default_factory=list)


@dataclass
class MetricsView:
    """A materialized, independent snapshot of persisted metrics.

    `scalars` maps each column to a per-iteration list (including ``iteration``
    itself), with nested columns such as ``memory`` regrouped into sub-dicts.
    `evaluations` maps eval type (``training``/``checkpoint``) to label to the
    sparse win/loss/draw points recorded for that eval.
    """
    scalars: dict[str, Any]
    evaluations: dict[str, dict[str, EvalSeries]]


class MetricsStore:

    def __init__(
        self,
        metrics_dir: Path,
        label_types: dict[str, str],
        resume: bool,
        truncate_to: Optional[int] = None,
    ) -> None:
        self._metrics_dir = metrics_dir
        self._label_types = dict(label_types)
        self._scalars_path = metrics_dir / "scalars.csv"
        self._evals_path = metrics_dir / "evals.csv"
        self._evals: dict[str, EvalSeries] = {label: EvalSeries() for label in label_types}

        metrics_dir.mkdir(parents=True, exist_ok=True)
        if truncate_to is not None:
            MetricsCSVWriter.truncate_to_iteration(self._scalars_path, truncate_to)
            MetricsCSVWriter.truncate_to_iteration(self._evals_path, truncate_to)

        self._scalars_writer = MetricsCSVWriter(self._scalars_path, resume=resume)
        self._evals_writer = MetricsCSVWriter(self._evals_path, resume=resume)
        if resume:
            self._load_evals()

    def record_step(self, iteration: int, scalars: dict[str, Any]) -> None:
        """Append one dense row of per-iteration scalar metrics to disk."""
        self._scalars_writer.append(iteration, self._flatten_row(scalars))

    def record_eval(
        self, iteration: int, label: str, position: str, wins: int, losses: int, draws: int
    ) -> None:
        """Record one evaluation result, both in memory and appended to disk."""
        point = EvalPoint(iteration=iteration, wins=wins, losses=losses, draws=draws)
        series = self._evals.setdefault(label, EvalSeries())
        if position == "as_p0":
            series.as_p0.append(point)
        elif position == "as_p1":
            series.as_p1.append(point)
        else:
            raise ValueError(f"Unknown eval position: {position!r}")
        self._evals_writer.append(iteration, {
            "eval_type": self._label_types.get(label, ""),
            "label": label,
            "position": position,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        })

    def load_view(
        self,
        max_points: Optional[int] = None,
        up_to: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> MetricsView:
        """Build a `MetricsView` from disk, streaming the scalar file.

        `up_to` keeps only iterations at or before a cutoff. `tail` keeps only
        the last N scalar rows. `max_points` downsamples the scalar rows to at
        most N evenly-spaced points. Evaluation points are sparse and always
        returned in full (subject to `up_to`).
        """
        rows = self._read_scalar_rows(up_to=up_to, max_points=max_points, tail=tail)
        return MetricsView(
            scalars=self._rows_to_scalars(rows),
            evaluations=self._eval_view(up_to=up_to),
        )

    def truncate_to(self, iteration: int) -> None:
        """Drop every persisted row past `iteration`, in memory and on disk."""
        MetricsCSVWriter.truncate_to_iteration(self._scalars_path, iteration)
        MetricsCSVWriter.truncate_to_iteration(self._evals_path, iteration)
        for series in self._evals.values():
            series.as_p0 = [point for point in series.as_p0 if point.iteration <= iteration]
            series.as_p1 = [point for point in series.as_p1 if point.iteration <= iteration]

    def close(self) -> None:
        self._scalars_writer.close()
        self._evals_writer.close()

    @staticmethod
    def _flatten_row(scalars: dict[str, Any]) -> dict[str, Any]:
        """Flatten one level of nesting into ``parent.child`` columns, dropping
        the evaluations bucket (persisted separately)."""
        flat: dict[str, Any] = {}
        for key, value in scalars.items():
            if key in ("iteration", "evaluations"):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        return flat

    @staticmethod
    def _nest_columns(columns: dict[str, list]) -> dict[str, Any]:
        """Regroup ``parent.child`` columns back into nested sub-dicts."""
        nested: dict[str, Any] = {}
        for key, series in columns.items():
            if "." in key:
                parent, child = key.split(".", 1)
                nested.setdefault(parent, {})[child] = series
            else:
                nested[key] = series
        return nested

    @staticmethod
    def _parse_cell(value: str) -> Optional[float | str]:
        if value == "" or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def _read_scalar_rows(
        self, up_to: Optional[int], max_points: Optional[int], tail: Optional[int]
    ) -> list[dict[str, str]]:
        if not self._scalars_path.exists():
            return []
        if tail is not None:
            return self._read_tail_rows(up_to, tail)
        if max_points is not None:
            return self._read_downsampled_rows(up_to, max_points)
        return list(self._iter_rows(up_to))

    def _iter_rows(self, up_to: Optional[int]) -> Iterator[dict[str, str]]:
        with open(self._scalars_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if up_to is not None and int(row["iteration"]) > up_to:
                    continue
                yield row

    def _read_tail_rows(self, up_to: Optional[int], tail: int) -> list[dict[str, str]]:
        kept: deque[dict[str, str]] = deque(maxlen=tail)
        for row in self._iter_rows(up_to):
            kept.append(row)
        return list(kept)

    def _read_downsampled_rows(self, up_to: Optional[int], max_points: int) -> list[dict[str, str]]:
        total = sum(1 for _ in self._iter_rows(up_to))
        if total <= max_points:
            return list(self._iter_rows(up_to))
        if max_points <= 1:
            return [row for index, row in enumerate(self._iter_rows(up_to)) if index == total - 1]
        keep_indices = {round(k * (total - 1) / (max_points - 1)) for k in range(max_points)}
        return [row for index, row in enumerate(self._iter_rows(up_to)) if index in keep_indices]

    def _rows_to_scalars(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        if not rows:
            return {"iteration": []}
        field_names = [name for name in rows[0].keys() if name is not None]
        columns: dict[str, list] = {name: [] for name in field_names}
        for row in rows:
            for name in field_names:
                cell = row.get(name, "")
                if name == "iteration":
                    columns[name].append(int(cell))
                else:
                    columns[name].append(self._parse_cell(cell))
        return self._nest_columns(columns)

    def _eval_view(self, up_to: Optional[int]) -> dict[str, dict[str, EvalSeries]]:
        view: dict[str, dict[str, EvalSeries]] = {}
        for label, series in self._evals.items():
            eval_type = self._label_types.get(label)
            if eval_type is None:
                continue
            view.setdefault(eval_type, {})[label] = EvalSeries(
                as_p0=self._filter_points(series.as_p0, up_to),
                as_p1=self._filter_points(series.as_p1, up_to),
            )
        return view

    def _filter_points(self, points: list[EvalPoint], up_to: Optional[int]) -> list[EvalPoint]:
        if up_to is None:
            return list(points)
        return [point for point in points if point.iteration <= up_to]

    def _load_evals(self) -> None:
        if not self._evals_path.exists():
            return
        with open(self._evals_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                series = self._evals.get(row.get("label"))
                if series is None:
                    continue
                point = EvalPoint(
                    iteration=int(row["iteration"]),
                    wins=int(row["wins"]),
                    losses=int(row["losses"]),
                    draws=int(row["draws"]),
                )
                position = row.get("position")
                if position == "as_p0":
                    series.as_p0.append(point)
                elif position == "as_p1":
                    series.as_p1.append(point)
