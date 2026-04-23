from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("alphazoo")

_SCALAR_TYPES = (int, float, bool, str)


class MetricsCSVWriter:
    """Append-only writer that locks the column schema at first write.

    On the first ``append`` call (fresh run) the header is written from that
    row's keys. On resume, the header is read from the existing file and the
    same lock applies. Keys appearing later that weren't in the header emit
    a one-time warning and are dropped; missing keys yield blank cells.
    """

    def __init__(self, path: Path, resume: bool) -> None:
        self._path = path
        self._header: Optional[list[str]] = None
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._warned_keys: set[str] = set()

        if resume and path.exists():
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    self._header = next(reader)
                except StopIteration:
                    self._header = None
            self._file = open(path, "a", newline="")
            if self._header is not None:
                self._writer = csv.DictWriter(self._file, fieldnames=self._header, extrasaction="ignore")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", newline="")

    def append(self, iteration: int, scalars: dict[str, Any]) -> None:
        row: dict[str, Any] = {"iteration": iteration}
        for key, value in scalars.items():
            if key == "iteration":
                continue
            coerced = self._coerce_scalar(value)
            if coerced is None:
                continue
            row[key] = coerced

        if self._header is None:
            self._header = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._header, extrasaction="ignore")
            self._writer.writeheader()

        for key in list(row.keys()):
            if key not in self._header and key not in self._warned_keys:
                logger.warning(
                    "MetricsCSVWriter: dropping new column '%s' (header locked at first write)", key
                )
                self._warned_keys.add(key)

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def _coerce_scalar(self, value: Any) -> Optional[Union[str, int, float]]:
        """Return a CSV-safe scalar, or ``None`` if the value is non-scalar."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, _SCALAR_TYPES):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
