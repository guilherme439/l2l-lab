from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class CheckpointData:
    iteration: int
    metrics: Dict[str, List]


def get_checkpoint_path(model_dir: Path, iteration: Optional[int] = None) -> Optional[Path]:
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    checkpoints = list(checkpoints_dir.glob("model_iter_*.cp"))
    if not checkpoints:
        return None
    
    def get_iter(p: Path) -> int:
        return int(p.stem.split("_")[-1])
    
    if iteration is not None:
        target = checkpoints_dir / f"model_iter_{iteration}.cp"
        if target.exists():
            return target
        valid = [cp for cp in checkpoints if get_iter(cp) <= iteration]
        if valid:
            return max(valid, key=get_iter)
        return None
    
    return max(checkpoints, key=get_iter)


def get_latest_checkpoint_path(model_dir: Path) -> Optional[Path]:
    return get_checkpoint_path(model_dir, iteration=None)


def load_checkpoint_data(model_dir: Path, iteration: Optional[int] = None) -> Optional[CheckpointData]:
    cp_path = get_checkpoint_path(model_dir, iteration)
    if cp_path:
        cp_data = torch.load(cp_path, weights_only=False)
        return CheckpointData(
            iteration=cp_data.get("iteration", 0),
            metrics=cp_data.get("metrics", {}),
        )
    
    if iteration is None:
        model_cp = model_dir / "model.cp"
        if model_cp.exists():
            cp_data = torch.load(model_cp, weights_only=False)
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
