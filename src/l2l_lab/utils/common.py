from __future__ import annotations


def check_interval(iteration: int, interval: int) -> bool:
    """True when `iteration` is a positive multiple of `interval`.
    A non-positive `interval` is treated as "never fire"."""
    if interval <= 0:
        return False
    return iteration % interval == 0
