import errno
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from l2l_lab.utils.common import find_paths_with_iteration_past
import logging

logger = logging.getLogger("l2l_lab")

_CHECKPOINT_DIR_PATTERN = re.compile(r"^(\d+)$")


def get_temp_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    temp_dir = Path(cache_root) / "l2l_lab" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def atomic_write(path: Path, write_to_temp: Callable[[Path], None]) -> None:
    """Write to a temp file in l2l-lab's own temp directory, then rename it onto `path`.

    The rename is atomic when the temp directory and `path` share a filesystem. When they do
    not, `os.replace` raises `EXDEV`; the write then falls back to a non-atomic move.
    """
    fd, temp_name = tempfile.mkstemp(dir=get_temp_dir())
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_to_temp(temp_path)
        try:
            os.replace(temp_path, path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            logger.warning(f"Directory is on a different filesystem; using non-atomic write to '{path}'")
            shutil.move(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


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


def list_checkpoint_iterations(model_dir: Path) -> list[int]:
    """Return the iteration numbers of all checkpoint directories, sorted ascending."""
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return sorted(int(d.name) for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit())


def get_checkpoint_dir(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    iterations = list_checkpoint_iterations(model_dir)
    if iteration is not None:
        iterations = [it for it in iterations if it <= iteration]
    if not iterations:
        return None
    return model_dir / "checkpoints" / str(max(iterations))


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


def is_rewind(model_dir: Path, loaded_iteration: int) -> bool:
    """True when `loaded_iteration` falls behind the highest checkpoint on disk."""
    latest_dir = get_latest_checkpoint_dir(model_dir)
    if latest_dir is None:
        return False
    try:
        latest_iter = int(latest_dir.name)
    except ValueError:
        return False
    return loaded_iteration < latest_iter


def trim_metrics_to_iteration(metrics: dict[str, list], target_iteration: int) -> dict[str, list]:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return metrics

    try:
        cutoff_idx = next(i for i, it in enumerate(iterations) if it > target_iteration)
    except StopIteration:
        return metrics

    return {k: v[:cutoff_idx] for k, v in metrics.items()}
