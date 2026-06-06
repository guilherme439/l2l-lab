import re
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from l2l_lab.utils.common import find_paths_with_iteration_past
import logging

logger = logging.getLogger("l2l_lab")

_CHECKPOINT_DIR_PATTERN = re.compile(r"^(\d+)$")


def load_checkpoint_file(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def load_model_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    """Load `state_dict` into `model`, falling back to non-strict on architecture mismatch.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        yellow_color_tags = "\033[33m", "\033[0m"
        start_tag, end_tag = yellow_color_tags
        logger.warning(
            f"{start_tag}\n"
            "WARNING: Strict load_state_dict failed — network architecture has "
            "changed since this checkpoint was saved. Falling back to "
            f"non-strict load.{end_tag} Original error: \n{exc}\n"
        )

    model.load_state_dict(state_dict, strict=False)


def get_checkpoint_dir(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not checkpoint_dirs:
        return None

    if iteration is not None:
        target = checkpoints_dir / str(iteration)
        if target.exists():
            return target
        valid = [d for d in checkpoint_dirs if int(d.name) <= iteration]
        if valid:
            return max(valid, key=lambda d: int(d.name))
        return None

    return max(checkpoint_dirs, key=lambda d: int(d.name))


def get_algo_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / "algo"


def get_training_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / "training.cp"


def get_latest_checkpoint_dir(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_dir(model_dir, iteration=None)


def load_trainer_checkpoint(model_dir: Path, iteration: Optional[int] = None) -> Optional[dict[str, Any]]:
    cp_path = get_training_checkpoint_path(model_dir, iteration)
    if cp_path is None or not cp_path.exists():
        return None
    return load_checkpoint_file(cp_path)


def list_checkpoint_iterations_past(model_dir: Path, iteration: int) -> list[int]:
    """Return iteration numbers of checkpoint directories with iter > `iteration`, sorted ascending."""
    matches = find_paths_with_iteration_past(
        model_dir / "checkpoints", _CHECKPOINT_DIR_PATTERN, iteration,
    )
    return sorted(it for _, it in matches)


def delete_checkpoint_dirs_past(model_dir: Path, iteration: int) -> None:
    """Remove every ``models/<name>/checkpoints/<N>/`` directory with N > `iteration`."""
    matches = find_paths_with_iteration_past(
        model_dir / "checkpoints", _CHECKPOINT_DIR_PATTERN, iteration,
    )
    for path, _ in matches:
        if path.is_dir():
            shutil.rmtree(path)


def is_rewind(model_dir: Path, start_iteration: int) -> bool:
    """True when `start_iteration` falls behind the highest checkpoint on disk."""
    latest_dir = get_latest_checkpoint_dir(model_dir)
    if latest_dir is None:
        return False
    try:
        latest_iter = int(latest_dir.name)
    except ValueError:
        return False
    return start_iteration < latest_iter


def trim_metrics_to_iteration(metrics: dict[str, list], target_iteration: int) -> dict[str, list]:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return metrics

    try:
        cutoff_idx = next(i for i, it in enumerate(iterations) if it > target_iteration)
    except StopIteration:
        return metrics

    return {k: v[:cutoff_idx] for k, v in metrics.items()}
