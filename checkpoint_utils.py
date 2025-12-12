from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class CheckpointData:
    iteration: int
    metrics: Dict[str, List]


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


def get_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir:
        return checkpoint_dir / "model.cp"
    return None


def get_algo_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoint_dir = get_checkpoint_dir(model_dir, iteration)
    if checkpoint_dir:
        return checkpoint_dir / "algo_checkpoint"
    return None


def get_latest_checkpoint_path(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_path(model_dir, iteration=None)


def get_latest_checkpoint_dir(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_dir(model_dir, iteration=None)


def load_checkpoint_data(model_dir: Path, iteration: Optional[int] = None) -> Optional[CheckpointData]:
    cp_path = get_checkpoint_path(model_dir, iteration)
    if cp_path and cp_path.exists():
        cp_data = torch.load(cp_path, weights_only=False)
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
