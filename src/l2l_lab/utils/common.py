from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def check_interval(iteration: int, interval: int) -> bool:
    """True when `iteration` is a positive multiple of `interval`.
    A non-positive `interval` is treated as "never fire"."""
    if interval <= 0:
        return False
    return iteration % interval == 0


def clone_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """
        Return a shallow-per-key deep-copy of a PettingZoo observation dict.
    """
    cloned: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            cloned[k] = v.copy()
        else:
            cloned[k] = v
    return cloned


def find_paths_with_iteration_past(
    parent: Path,
    pattern: re.Pattern[str],
    iteration: int,
) -> list[tuple[Path, int]]:
    """Scan `parent` for entries whose names match `pattern` and whose first
    captured group parses as an integer strictly greater than `iteration`.

    Returns ``(path, iteration)`` pairs. The parent not existing is treated as
    "nothing matches".
    """
    if not parent.exists():
        return []
    matches: list[tuple[Path, int]] = []
    for path in parent.iterdir():
        m = pattern.match(path.name)
        if m is None:
            continue
        parsed = int(m.group(1))
        if parsed > iteration:
            matches.append((path, parsed))
    return matches
