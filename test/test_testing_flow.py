import os

from l2l_lab.testing.tester import Tester

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "testing_test.yml")


def test_testing_random_vs_random_completes() -> None:
    tester = Tester(CONFIG_PATH)
    results = tester.test()

    assert results.total == tester.config.num_games
    assert results.wins + results.losses + results.draws == results.total
    assert results.elapsed_time >= 0.0
