"""Shared assertions and utilities for end-to-end training tests. Each test builds
and runs its own `Trainer`, then calls these to check the outcome."""

import shutil
from pathlib import Path

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
    iterations = trainer.load_metrics().scalars["iteration"]
    assert len(iterations) == expected_iterations, (
        f"expected {expected_iterations} iterations, got {len(iterations)}: {iterations}"
    )
    assert checkpoint_iterations(trainer.config.name), "no checkpoint was written"


def assert_eval_results(view, category: str) -> None:
    series_by_label = view.evaluations.get(category, {})
    assert series_by_label, f"no '{category}' evaluations were recorded"
    has_result = any(
        series.as_p0 or series.as_p1
        for series in series_by_label.values()
    )
    assert has_result, f"'{category}' evaluations recorded no game results"
