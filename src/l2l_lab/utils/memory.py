"""Memory sampling for training runs.

Tracks per-process-tree RAM usage. Both Ray RLlib (raylet + env-runner workers)
and AlphaZoo (MCTS workers) spawn their workers as descendants of the Python
process running the trainer, so a recursive child walk reaches them all.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import psutil

logger = logging.getLogger("alphazoo")

_BYTES_PER_MB = 1024 * 1024


@dataclass
class MemorySample:
    main_pss_mb: float
    workers_pss_mb: float
    total_pss_mb: float


class MemorySampler:
    """Samples PSS RAM usage of the trainer process and its descendants.

    PSS (proportional set size) divides each shared page by the number of
    processes sharing it, so the per-tree sum reflects true RAM cost rather
    than the inflated number you'd get summing RSS across forked workers.
    Falls back to RSS if PSS is unavailable (non-Linux platforms).
    """

    def __init__(self) -> None:
        self._main = psutil.Process(os.getpid())
        self._pss_supported = self._detect_pss_support()
        if not self._pss_supported:
            logger.warning(
                "PSS not available on this platform — memory tracking will fall back "
                "to RSS, which double-counts pages shared between forked workers."
            )

    def sample(self) -> MemorySample:
        main_pss_mb = self._safe_proc_mb(self._main)
        workers_pss_mb = sum(self._safe_proc_mb(c) for c in self._safe_children())
        return MemorySample(
            main_pss_mb=main_pss_mb,
            workers_pss_mb=workers_pss_mb,
            total_pss_mb=main_pss_mb + workers_pss_mb,
        )

    def _detect_pss_support(self) -> bool:
        try:
            info = self._main.memory_full_info()
        except (psutil.AccessDenied, AttributeError):
            return False
        return hasattr(info, "pss")

    def _safe_children(self) -> list[psutil.Process]:
        try:
            return self._main.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _safe_proc_mb(self, proc: psutil.Process) -> float:
        try:
            if self._pss_supported:
                return proc.memory_full_info().pss / _BYTES_PER_MB
            return proc.memory_info().rss / _BYTES_PER_MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
