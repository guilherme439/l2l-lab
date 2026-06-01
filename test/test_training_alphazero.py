import os

from _train_utils import assert_resume_extends_training, assert_training_completes

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_alphazero_resnet_trains() -> None:
    """AlphaZero + ResNet + reanalyse, with MCTS training and checkpoint evals on tictactoe."""
    assert_training_completes(
        os.path.join(CONFIG_DIR, "alphazero_resnet_tictactoe.yml"),
        run_name="_test_alphazero_resnet",
        check_training_eval=True,
        check_checkpoint_eval=True,
    )


def test_alphazero_snnet_resumes() -> None:
    """AlphaZero + SNNet with epochs/SGD resumes from a checkpoint, with policy evals."""
    assert_resume_extends_training(
        os.path.join(CONFIG_DIR, "alphazero_snnet_resume_init.yml"),
        os.path.join(CONFIG_DIR, "alphazero_snnet_resume_continue.yml"),
        run_name="_test_alphazero_snnet_resume",
        check_training_eval=True,
    )
