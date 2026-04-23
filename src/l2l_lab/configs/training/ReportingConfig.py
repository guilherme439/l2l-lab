from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReportingConfig:
    """Diagnostic reporting knobs.

    When ``enabled`` is False, the Reporter becomes a no-op and no files are
    written. When enabled, per-iteration scalars stream to a CSV and a heavy
    Markdown snapshot is emitted every ``interval`` iterations.
    """
    enabled: bool = False
    interval: int = 100
    sample_games_per_eval: int = 2
