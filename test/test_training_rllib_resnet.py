import os
import shutil
from pathlib import Path

from l2l_lab.training.trainer import Trainer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "rllib_resnet_test.yml")


def test_rllib_resnet_training_completes() -> None:
    """
        RLlib PPO + ResNet + tictactoe.
    """
    trainer = Trainer(CONFIG_PATH)
    try:
        trainer.train()

        assert trainer.metrics["iteration"], "No iterations recorded"
        evaluations = trainer.metrics.get("evaluations", {})
        assert "policy_vs_random" in evaluations
        assert "policy_vs_policy" in evaluations
        assert any(w is not None for w in evaluations["policy_vs_random"]["as_p0"]["wins"])
        assert any(w is not None for w in evaluations["policy_vs_random"]["as_p1"]["wins"])
    finally:
        model_dir = Path("models") / trainer.config.name
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
