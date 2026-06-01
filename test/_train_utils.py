import shutil
from pathlib import Path
from typing import Any

from l2l_lab.training.trainer import Trainer

_MODELS_DIR = Path("models")


def assert_training_completes(
    config_path: str,
    run_name: str,
    check_training_eval: bool = False,
    check_checkpoint_eval: bool = False,
) -> None:
    _remove_run(run_name)
    trainer = Trainer(config_path)
    try:
        trainer.train()

        expected_iterations = trainer.config.backend.algorithm.total_iterations
        iterations = trainer.metrics["iteration"]
        assert len(iterations) == expected_iterations, (
            f"expected {expected_iterations} iterations, got {len(iterations)}: {iterations}"
        )

        checkpoints = _MODELS_DIR / run_name / "checkpoints"
        assert checkpoints.exists() and any(checkpoints.iterdir()), "no checkpoint was written"

        if check_training_eval:
            _assert_eval_results(trainer.metrics, "training")
        if check_checkpoint_eval:
            _assert_eval_results(trainer.metrics, "checkpoint")
    finally:
        _remove_run(run_name)


def assert_resume_extends_training(
    init_config: str,
    continue_config: str,
    run_name: str,
    check_training_eval: bool = False,
) -> None:
    _remove_run(run_name)
    try:
        initial = Trainer(init_config)
        initial.train()
        initial_iterations = list(initial.metrics["iteration"])
        assert initial_iterations, "initial run recorded no iterations"

        checkpoints = _MODELS_DIR / run_name / "checkpoints"
        assert checkpoints.exists() and any(checkpoints.iterdir()), "initial run wrote no checkpoint to resume from"

        resumed = Trainer(continue_config)
        resumed.train()
        resumed_iterations = list(resumed.metrics["iteration"])

        assert len(resumed_iterations) > len(initial_iterations), (
            f"resume did not extend training: initial={initial_iterations}, resumed={resumed_iterations}"
        )
        assert resumed_iterations[-1] > initial_iterations[-1], (
            f"resume did not progress past the initial run: initial={initial_iterations}, resumed={resumed_iterations}"
        )

        if check_training_eval:
            _assert_eval_results(resumed.metrics, "training")
    finally:
        _remove_run(run_name)


def _assert_eval_results(metrics: dict[str, Any], category: str) -> None:
    buckets = metrics.get("evaluations", {}).get(category, {})
    assert buckets, f"no '{category}' evaluations were recorded"
    has_result = any(
        value is not None
        for bucket in buckets.values()
        for position in ("as_p0", "as_p1")
        for value in bucket.get(position, {}).get("wins", [])
    )
    assert has_result, f"'{category}' evaluations recorded no game results"


def _remove_run(run_name: str) -> None:
    run_dir = _MODELS_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
