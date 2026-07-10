import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import wandb
import yaml

from l2l_lab.configs.training.training_config import TrainingConfig
import logging

logger = logging.getLogger("l2l_lab")

_APPLICATION_CONFIG_PATH = Path("application.yml")
_API_KEY_PLACEHOLDER = "your-wandb-api-key-here"


class WandbUtils:

    @staticmethod
    def init(run_name: str, training_config: TrainingConfig) -> bool:
        try:
            logger.info("")
            wandb_settings = WandbUtils._load_wandb_settings()
            if wandb_settings is None:
                return False
            if not wandb_settings.get("enabled", False):
                logger.info(f"wandb: disabled in {_APPLICATION_CONFIG_PATH}")
                return False

            api_key = wandb_settings.get("api_key")
            if not api_key or api_key == _API_KEY_PLACEHOLDER:
                logger.warning(f"WARNING: wandb: api_key missing or placeholder in {_APPLICATION_CONFIG_PATH}; skipping")
                return False

            os.environ["WANDB_API_KEY"] = api_key
            os.environ.setdefault("WANDB_HTTP_TIMEOUT", "60")

            config_dict = WandbUtils.extract_hyperparameters(training_config)

            init_kwargs: dict[str, Any] = {
                "project": wandb_settings.get("project", "l2l-lab"),
                "entity": wandb_settings.get("entity"),
                "name": run_name,
                "group": run_name,
                "config": config_dict,
                "tags": wandb_settings.get("tags") or [],
                "settings": wandb.Settings(
                    init_timeout=180,
                    _service_wait=300,
                ),
            }

            run = wandb.init(**init_kwargs)

            logger.info("")
            if run is None:
                logger.warning("WARNING: wandb: init returned None; skipping")
                return False

            # Evaluations complete out of order with training steps, so they log on
            # their own x-axis instead of the (monotonically increasing) step used by `log`
            run.define_metric("evaluations/*", step_metric="eval_iteration")

            logger.info(f"wandb: run started (id={run.id}, project={wandb_settings.get('project')})")
            return True
        except Exception as e:
            logger.warning(f"WARNING: wandb: init failed ({e}); training will continue without wandb")
            return False

    @staticmethod
    def log(metrics: dict[str, Any], step: int) -> None:
        try:
            flat = WandbUtils._flatten(metrics)
            if not flat:
                return
            wandb.log(flat, step=step)
        except Exception as e:
            logger.warning(f"WARNING: wandb: log failed ({e})")

    @staticmethod
    def log_evaluations(metrics: dict[str, Any], iteration: int) -> None:
        """Log evaluation results against the `eval_iteration` axis `init` defined
        for `evaluations/*`, independent of the training step counter `log` uses.
        """
        try:
            flat = WandbUtils._flatten(metrics)
            if not flat:
                return
            flat["eval_iteration"] = iteration
            wandb.log(flat)
        except Exception as e:
            logger.warning(f"WARNING: wandb: log_evaluations failed ({e})")

    @staticmethod
    def finish() -> None:
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"WARNING: wandb: finish failed ({e})")

    @staticmethod
    def extract_hyperparameters(training_config: TrainingConfig) -> dict[str, Any]:
        try:
            return asdict(training_config)
        except Exception as e:
            logger.warning(f"WARNING: wandb: failed to convert TrainingConfig ({e}); using minimal config")
            return {"_config_name": training_config.name}

    @staticmethod
    def _load_wandb_settings() -> Optional[dict[str, Any]]:
        if not _APPLICATION_CONFIG_PATH.exists():
            logger.info(f"wandb: {_APPLICATION_CONFIG_PATH} not found; wandb disabled")
            return None
        with open(_APPLICATION_CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        return data.get("wandb")

    @staticmethod
    def _flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in metrics.items():
            if key == "iteration":
                continue
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(WandbUtils._flatten(value, full_key))
            elif value is not None:
                flat[full_key] = value
        return flat
