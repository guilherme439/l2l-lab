import os
import shutil
from pathlib import Path

from l2l_lab.training.trainer import Trainer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "alphazoo_tictactoe_test.yml")


def test_alphazoo_tictactoe_training_completes() -> None:
    """
        AlphaZoo + MLPNet + tictactoe + inference cache + Adam + samples.
    """
    trainer = Trainer(CONFIG_PATH)
    try:
        trainer.train()

        iterations = trainer.metrics["iteration"]
        assert iterations, "No iterations recorded"

        evaluations = trainer.metrics.get("evaluations", {})
        assert "mcts_vs_random" in evaluations
        assert "policy_vs_mcts" in evaluations
        for bucket in evaluations.values():
            for position in ("as_p0", "as_p1"):
                assert len(bucket[position]["wins"]) == len(iterations)

        # training_eval fires at iter 5 and 10
        assert any(w is not None for w in evaluations["mcts_vs_random"]["as_p0"]["wins"])
        assert any(w is not None for w in evaluations["mcts_vs_random"]["as_p1"]["wins"])
        # checkpoint_eval with mcts opponent needs a previous checkpoint → fires at iter 10
        assert any(w is not None for w in evaluations["policy_vs_mcts"]["as_p0"]["wins"])
        assert any(w is not None for w in evaluations["policy_vs_mcts"]["as_p1"]["wins"])

        cp_root = Path("models") / trainer.config.name / "checkpoints"
        assert cp_root.exists() and any(cp_root.iterdir())
    finally:
        _cleanup(trainer)


def _cleanup(trainer: Trainer) -> None:
    model_dir = Path("models") / trainer.config.name
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
