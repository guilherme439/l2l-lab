import os

from _train_utils import assert_eval_results, assert_run_completed, checkpoint_iterations
from l2l_lab.training.trainer import Trainer

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def test_alphazero_resnet_trains(clean_test_model_dirs) -> None:
    """AlphaZero + ResNet + reanalyse, with MCTS training and checkpoint evals on tictactoe."""
    trainer = Trainer(os.path.join(CONFIG_DIR, "alphazero_resnet_tictactoe.yml"))
    trainer.train()
    assert_run_completed(trainer)
    assert_eval_results(trainer.load_metrics(), "training")
    assert_eval_results(trainer.load_metrics(), "checkpoint")


def test_alphazero_snnet_resumes(clean_test_model_dirs) -> None:
    """AlphaZero + SNNet with epochs/SGD resumes from a checkpoint, with policy evals."""
    initial = Trainer(os.path.join(CONFIG_DIR, "alphazero_snnet_resume_init.yml"))
    initial.train()
    initial_iterations = list(initial.load_metrics().scalars["iteration"])

    resumed = Trainer(os.path.join(CONFIG_DIR, "alphazero_snnet_resume_continue.yml"))
    resumed.train()
    resumed_iterations = list(resumed.load_metrics().scalars["iteration"])

    assert len(resumed_iterations) > len(initial_iterations), (
        f"resume did not extend training: initial={initial_iterations}, resumed={resumed_iterations}"
    )
    assert resumed_iterations[-1] > initial_iterations[-1], (
        f"resume did not progress past the initial run: initial={initial_iterations}, resumed={resumed_iterations}"
    )
    assert_eval_results(resumed.load_metrics(), "training")


def test_alphazero_snnet_rewinds(monkeypatch, clean_test_model_dirs) -> None:
    """AlphaZero + SNNet rewinds to an earlier checkpoint, discarding later iterations, then trains on."""
    monkeypatch.setattr("l2l_lab.training.trainer._DESTRUCTIVE_ABORT_WAIT_SECONDS", 0)
    run_name = "_test_alphazero_snnet_rewind"
    rewind_to = 3

    initial = Trainer(os.path.join(CONFIG_DIR, "alphazero_snnet_rewind_init.yml"))
    initial.train()
    initial_iterations = list(initial.load_metrics().scalars["iteration"])
    initial_checkpoints = checkpoint_iterations(run_name)
    assert any(iteration > rewind_to for iteration in initial_checkpoints), (
        f"initial run needs a checkpoint past {rewind_to} to exercise rewind: {sorted(initial_checkpoints)}"
    )

    rewound = Trainer(os.path.join(CONFIG_DIR, "alphazero_snnet_rewind_continue.yml"))
    rewound.train()
    rewound_iterations = list(rewound.load_metrics().scalars["iteration"])

    assert rewound_iterations == list(range(len(rewound_iterations))), (
        f"rewound metrics are not contiguous from 0: {rewound_iterations}"
    )
    assert rewound_iterations[-1] < initial_iterations[-1], (
        f"rewind did not discard later iterations: initial={initial_iterations}, rewound={rewound_iterations}"
    )

    final_iteration = rewound_iterations[-1]
    remaining_checkpoints = checkpoint_iterations(run_name)
    assert all(iteration <= final_iteration for iteration in remaining_checkpoints), (
        f"checkpoints past the final rewound iteration survived: {sorted(remaining_checkpoints)}"
    )
    stale_checkpoints = {iteration for iteration in initial_checkpoints if iteration > final_iteration}
    assert not (stale_checkpoints & remaining_checkpoints), (
        f"stale checkpoints from the initial run were not removed: {sorted(stale_checkpoints & remaining_checkpoints)}"
    )
