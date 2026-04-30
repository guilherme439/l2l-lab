from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class CheckpointData:
    iteration: int
    metrics: Dict[str, List]


def load_checkpoint_file(path: Path) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(path, weights_only=False, map_location=device)


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


def _find_checkpoint_file(checkpoint_dir: Path) -> Optional[Path]:
    training_dir = checkpoint_dir / "training"
    for name in ("checkpoint.pt", "data.pt"):
        p = training_dir / name
        if p.exists():
            return p
    return None


def get_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir:
        return _find_checkpoint_file(checkpoint_dir)
    return None


def get_algo_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / "training" / "algo_checkpoint"


def get_latest_checkpoint_path(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_path(model_dir, iteration=None)


def get_latest_checkpoint_dir(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_dir(model_dir, iteration=None)


def load_checkpoint_data(model_dir: Path, iteration: Optional[int] = None) -> Optional[CheckpointData]:
    cp_path = get_checkpoint_path(model_dir, iteration)
    if cp_path and cp_path.exists():
        cp_data = load_checkpoint_file(cp_path)
        return CheckpointData(
            iteration=cp_data.get("iteration", 0),
            metrics=cp_data.get("metrics", {}),
        )
    return None


def trim_metrics_to_iteration(metrics: Dict[str, List], target_iteration: int) -> Dict[str, List]:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return metrics

    try:
        cutoff_idx = next(i for i, it in enumerate(iterations) if it > target_iteration)
    except StopIteration:
        return metrics

    return {k: v[:cutoff_idx] for k, v in metrics.items()}
