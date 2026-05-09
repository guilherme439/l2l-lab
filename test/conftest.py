import pytest


@pytest.fixture(autouse=True)
def _disable_wandb(monkeypatch):
    from l2l_lab.utils import wandb as wandb_helper
    monkeypatch.setattr(wandb_helper, "_load_wandb_settings", lambda: None)
