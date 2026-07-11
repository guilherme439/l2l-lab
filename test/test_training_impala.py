import os

from _train_utils import assert_eval_results, assert_run_completed
from l2l_lab.training.trainer import Trainer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_impala_mlpnet_trains(clean_test_model_dirs) -> None:
    """IMPALA + MLPNet trains on tictactoe, with policy training evals."""
    trainer = Trainer(os.path.join(CONFIG_DIR, "impala_mlpnet_tictactoe.yml"))
    trainer.train()
    assert_run_completed(trainer)
    assert_eval_results(trainer.load_metrics(), "training")
