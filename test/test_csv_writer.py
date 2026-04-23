from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from l2l_lab.reporting.csv_writer import MetricsCSVWriter


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def test_header_locks_at_first_write(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)

    writer.append(1, {"loss": 0.5, "reward": 1.0})
    writer.append(2, {"loss": 0.4, "reward": 1.1, "new_metric": 42.0})
    writer.close()

    rows = _read_rows(path)
    assert list(rows[0].keys()) == ["iteration", "loss", "reward"]
    assert rows[0]["loss"] == "0.5"
    assert len(rows) == 2
    assert "new_metric" not in rows[1]


def test_missing_keys_yield_blank_cells(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)

    writer.append(1, {"loss": 0.5, "reward": 1.0})
    writer.append(2, {"loss": 0.4})
    writer.close()

    rows = _read_rows(path)
    assert rows[1]["reward"] == ""


def test_flat_scalar_filter_drops_dicts_and_lists(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)

    writer.append(1, {
        "loss": 0.5,
        "evaluations": {"foo": [1, 2, 3]},
        "values": [1.0, 2.0],
    })
    writer.close()

    rows = _read_rows(path)
    assert list(rows[0].keys()) == ["iteration", "loss"]


def test_numpy_scalars_coerce_to_float(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)

    writer.append(1, {"loss": np.float32(0.25), "count": np.int64(7)})
    writer.close()

    rows = _read_rows(path)
    assert float(rows[0]["loss"]) == 0.25
    assert float(rows[0]["count"]) == 7.0


def test_resume_preserves_header_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)
    writer.append(1, {"loss": 0.5, "reward": 1.0})
    writer.close()

    resumed = MetricsCSVWriter(path, resume=True)
    resumed.append(2, {"loss": 0.4, "reward": 1.1, "new_metric": 99.0})
    resumed.close()

    rows = _read_rows(path)
    assert list(rows[0].keys()) == ["iteration", "loss", "reward"]
    assert len(rows) == 2
    assert rows[1]["loss"] == "0.4"
    assert "new_metric" not in rows[1]


def test_none_values_are_blank(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)
    writer.append(1, {"loss": 0.5, "note": None})
    writer.close()

    rows = _read_rows(path)
    assert rows[0]["note"] == ""


def test_bool_coerces_to_int(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    writer = MetricsCSVWriter(path, resume=False)
    writer.append(1, {"converged": True, "stalled": False})
    writer.close()

    rows = _read_rows(path)
    assert rows[0]["converged"] == "1"
    assert rows[0]["stalled"] == "0"
