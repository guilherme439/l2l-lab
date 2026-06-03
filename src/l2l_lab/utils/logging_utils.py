from __future__ import annotations

import logging
import sys
from typing import TextIO


class GutterFormatter(logging.Formatter):
    """Prefixes every line of a record with a gutter bar, so l2l-lab's own output
    reads as a continuous left-hand column, distinct from other logs interleaved
    on the same stream. Blank lines carry the bare bar so the column stays
    unbroken."""

    _GUTTER = "▎ "

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        bar = self._GUTTER.rstrip()
        return "\n".join(
            f"{self._GUTTER}{line}" if line else bar
            for line in message.split("\n")
        )


class _DynamicStdoutHandler(logging.StreamHandler):
    """
    Writes each record to the current ``sys.stdout`` (resolved at emit time)
    rather than a stream captured at construction,
    so output passes through any stdout wrapper installed later
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:
        return sys.stdout
