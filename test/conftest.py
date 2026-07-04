from pathlib import Path

import pytest

from _train_utils import remove_run

_MODELS_DIR = Path("models")


@pytest.fixture(autouse=True)
def _disable_wandb(monkeypatch):
    from l2l_lab.utils import wandb as wandb_helper
    monkeypatch.setattr(wandb_helper, "_load_wandb_settings", lambda: None)


@pytest.fixture
def clean_test_model_dirs():
    _remove_test_model_dirs()
    yield
    _remove_test_model_dirs()


def _remove_test_model_dirs() -> None:
    if not _MODELS_DIR.exists():
        return
    for run_dir in _MODELS_DIR.glob("_test_*"):
        remove_run(run_dir.name)
