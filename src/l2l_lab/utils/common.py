from __future__ import annotations

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
