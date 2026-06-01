import os

from _train_utils import assert_training_completes

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_impala_mlpnet_trains() -> None:
    """IMPALA + MLPNet trains on tictactoe, with policy training evals."""
    assert_training_completes(
        os.path.join(CONFIG_DIR, "impala_mlpnet_tictactoe.yml"),
        run_name="_test_impala_mlpnet",
        check_training_eval=True,
    )
