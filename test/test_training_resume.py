import os
import shutil
from pathlib import Path

from l2l_lab.training.trainer import Trainer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
INIT_CONFIG = os.path.join(CONFIG_DIR, "alphazoo_tictactoe_resume_init.yml")
CONTINUE_CONFIG = os.path.join(CONFIG_DIR, "alphazoo_tictactoe_resume_continue.yml")


def test_alphazoo_training_resume() -> None:
    """Initial run trains 5 iters, second run resumes and trains up to 10."""
    run_name = "_test_alphazoo_tictactoe_resume"
    model_dir = Path("models") / run_name
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)

    try:
        initial = Trainer(INIT_CONFIG)
        initial.train()
        initial_len = len(initial.metrics["iteration"])
        assert initial_len == 5, f"expected 5 iters after initial run, got {initial_len}"

        cp_root = model_dir / "checkpoints"
        assert (cp_root / "5").exists(), "initial run should have saved a checkpoint at iter 5"

        resumed = Trainer(CONTINUE_CONFIG)
        resumed.train()
        resumed_len = len(resumed.metrics["iteration"])
        assert resumed_len == 10, f"expected 10 iters after resume, got {resumed_len}"
        assert resumed.metrics["iteration"][:initial_len] == initial.metrics["iteration"]
        assert (cp_root / "10").exists(), "resumed run should have saved a checkpoint at iter 10"
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
