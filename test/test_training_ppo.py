import os

from _train_utils import assert_eval_results, assert_run_completed
from l2l_lab.training.trainer import Trainer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_ppo_convnet_trains(clean_test_model_dirs) -> None:
    """PPO + ConvNet trains on tictactoe, with policy training and checkpoint evals."""
    trainer = Trainer(os.path.join(CONFIG_DIR, "ppo_convnet_tictactoe.yml"))
    trainer.train()
    assert_run_completed(trainer)
    assert_eval_results(trainer.load_metrics(), "training")
    assert_eval_results(trainer.load_metrics(), "checkpoint")


def test_ppo_mlpnet_resumes(clean_test_model_dirs) -> None:
    """PPO + MLPNet resumes from a checkpoint and keeps training, with policy evals."""
    initial = Trainer(os.path.join(CONFIG_DIR, "ppo_mlpnet_resume_init.yml"))
    initial.train()
    initial_iterations = list(initial.load_metrics().scalars["iteration"])

    resumed = Trainer(os.path.join(CONFIG_DIR, "ppo_mlpnet_resume_continue.yml"))
    resumed.train()
    resumed_iterations = list(resumed.load_metrics().scalars["iteration"])

    assert len(resumed_iterations) > len(initial_iterations), (
        f"resume did not extend training: initial={initial_iterations}, resumed={resumed_iterations}"
    )
    assert resumed_iterations[-1] > initial_iterations[-1], (
        f"resume did not progress past the initial run: initial={initial_iterations}, resumed={resumed_iterations}"
    )
    assert_eval_results(resumed.load_metrics(), "training")
