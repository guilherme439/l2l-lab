import os

from _train_utils import assert_resume_extends_training, assert_training_completes

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_ppo_convnet_trains() -> None:
    """PPO + ConvNet trains on tictactoe, with policy training and checkpoint evals."""
    assert_training_completes(
        os.path.join(CONFIG_DIR, "ppo_convnet_tictactoe.yml"),
        run_name="_test_ppo_convnet",
        check_training_eval=True,
        check_checkpoint_eval=True,
    )


def test_ppo_mlpnet_resumes() -> None:
    """PPO + MLPNet resumes from a checkpoint and keeps training, with policy evals."""
    assert_resume_extends_training(
        os.path.join(CONFIG_DIR, "ppo_mlpnet_resume_init.yml"),
        os.path.join(CONFIG_DIR, "ppo_mlpnet_resume_continue.yml"),
        run_name="_test_ppo_mlpnet_resume",
        check_training_eval=True,
    )
