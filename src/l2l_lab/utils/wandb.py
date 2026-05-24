from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import wandb as _wandb_pkg
import yaml

from l2l_lab.configs.training.TrainingConfig import TrainingConfig

_APPLICATION_CONFIG_PATH = Path("application.yml")
_RUN_STATE_FILENAME = "run_state.json"
_API_KEY_PLACEHOLDER = "your-wandb-api-key-here"


def init(
    run_name: str,
    training_config: TrainingConfig,
    model_dir: Path,
    resume: bool,
    is_rewind: bool,
) -> bool:
    try:
        print()
        wandb_settings = _load_wandb_settings()
        if wandb_settings is None:
            return False
        if not wandb_settings.get("enabled", False):
            print(f"wandb: disabled in {_APPLICATION_CONFIG_PATH}")
            return False

        api_key = wandb_settings.get("api_key")
        if not api_key or api_key == _API_KEY_PLACEHOLDER:
            print(f"WARNING: wandb: api_key missing or placeholder in {_APPLICATION_CONFIG_PATH}; skipping")
            return False

        os.environ["WANDB_API_KEY"] = api_key

        run_id: Optional[str] = None
        if resume:
            run_id = _read_persisted_run_id(model_dir)

        config_dict = extract_hyperparameters(training_config)

        init_kwargs: dict[str, Any] = {
            "project": wandb_settings.get("project", "l2l-lab"),
            "entity": wandb_settings.get("entity"),
            "name": run_name,
            "group": run_name,
            "config": config_dict,
            "tags": wandb_settings.get("tags") or [],
        }
        if resume and run_id is not None and not is_rewind:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        run = _wandb_pkg.init(**init_kwargs)
        if run is None:
            print("WARNING: wandb: init returned None; skipping")
            return False

        _persist_run_id(model_dir, run.id)
        print(f"wandb: run started (id={run.id}, project={wandb_settings.get('project')})")
        return True
    except Exception as e:
        print(f"WARNING: wandb: init failed ({e}); training will continue without wandb")
        return False


def log(metrics: dict[str, Any], step: int) -> None:
    try:
        flat = _flatten(metrics)
        if not flat:
            return
        _wandb_pkg.log(flat, step=step)
    except Exception as e:
        print(f"WARNING: wandb: log failed ({e})")


def finish() -> None:
    try:
        _wandb_pkg.finish()
    except Exception as e:
        print(f"WARNING: wandb: finish failed ({e})")


def extract_hyperparameters(training_config: TrainingConfig) -> dict[str, Any]:
    try:
        return asdict(training_config)
    except Exception as e:
        print(f"WARNING: wandb: failed to convert TrainingConfig ({e}); using minimal config")
        return {"_config_name": training_config.name}


def _load_wandb_settings() -> Optional[dict[str, Any]]:
    if not _APPLICATION_CONFIG_PATH.exists():
        print(f"wandb: {_APPLICATION_CONFIG_PATH} not found; wandb disabled")
        return None
    with open(_APPLICATION_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("wandb")


def _read_persisted_run_id(model_dir: Path) -> Optional[str]:
    state_path = model_dir / _RUN_STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        return data.get("wandb_run_id")
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: wandb: failed to read {state_path} ({e}); starting fresh run")
        return None


def _persist_run_id(model_dir: Path, run_id: str) -> None:
    state_path = model_dir / _RUN_STATE_FILENAME
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}
    state["wandb_run_id"] = run_id
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "iteration":
            continue
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        elif value is not None:
            flat[full_key] = value
    return flat
