"""Shared assertions and utilities for end-to-end training tests. Each test builds
and runs its own `Trainer`, then calls these to check the outcome."""

import shutil
from pathlib import Path
from typing import Any

_MODELS_DIR = Path("models")


def remove_run(run_name: str) -> None:
    run_dir = _MODELS_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def checkpoint_iterations(run_name: str) -> set[int]:
    checkpoints_dir = _MODELS_DIR / run_name / "checkpoints"
    if not checkpoints_dir.exists():
        return set()
    return {int(entry.name) for entry in checkpoints_dir.iterdir() if entry.is_dir() and entry.name.isdigit()}


def assert_run_completed(trainer) -> None:
    expected_iterations = trainer.config.backend.algorithm.total_iterations
    iterations = trainer.metrics["iteration"]
    assert len(iterations) == expected_iterations, (
        f"expected {expected_iterations} iterations, got {len(iterations)}: {iterations}"
    )
    assert checkpoint_iterations(trainer.config.name), "no checkpoint was written"


def assert_eval_results(metrics: dict[str, Any], category: str) -> None:
    buckets = metrics.get("evaluations", {}).get(category, {})
    assert buckets, f"no '{category}' evaluations were recorded"
    has_result = any(
        value is not None
        for bucket in buckets.values()
        for position in ("as_p0", "as_p1")
        for value in bucket.get(position, {}).get("wins", [])
    )
    assert has_result, f"'{category}' evaluations recorded no game results"
