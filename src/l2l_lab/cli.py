import argparse
import sys
import traceback
import warnings
from pathlib import Path

from l2l_lab.testing.tester import Tester
from l2l_lab.training.trainer import Trainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
    module=r"ray\.rllib\.utils\.metrics\.stats\.ema",
)


DEFAULT_TRAINING_CONFIG_PATH = "configs/training/ppo_training_config.example.yml"
DEFAULT_TESTING_CONFIG_PATH = "configs/testing/testing_config.example.yml"
PROFILE_OUTPUT_PATH = Path("profiling/profile_output.prof")


def main():
    parser = argparse.ArgumentParser(description="L2L Lab - Training and Testing")
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Run mode: 'train' for training, 'test' for checkpoint testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to standard paths based on mode)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable yappi profiling (main process, all threads) and save results to profile_output.prof"
    )

    args = parser.parse_args()

    if args.mode == "train":
        config_path = args.config or DEFAULT_TRAINING_CONFIG_PATH
        if args.profile:
            run_with_profiling(train, config_path)
        else:
            train(config_path)
    elif args.mode == "test":
        config_path = args.config or DEFAULT_TESTING_CONFIG_PATH
        if args.profile:
            run_with_profiling(test, config_path)
        else:
            test(config_path)

    print("\nDone!")


def train(config_path: str):
    try:
        trainer = Trainer(config_path)
        trainer.train()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()


def test(config_path: str):
    try:
        tester = Tester(config_path)
        tester.test()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()


def run_with_profiling(func, *args):
    import yappi

    yappi.set_clock_type("wall")
    yappi.start()

    try:
        func(*args)
    finally:
        yappi.stop()
        stats = yappi.get_func_stats()
        stats.save(str(PROFILE_OUTPUT_PATH), type="pstat")

        print("\n" + "=" * 70)
        print("PROFILING SUMMARY (top 20 by total time, all threads)")
        print("=" * 70)
        stats.sort("ttot", "desc")
        header = f"{'name':<50} {'ncall':>8} {'ttot':>8} {'tsub':>8} {'tavg':>8}"
        print(header)
        for stat in stats[:20]:
            print(f"{stat.full_name[:50]:<50} {stat.ncall:>8} {stat.ttot:>8.4f} {stat.tsub:>8.4f} {stat.tavg:>8.4f}")

        print("=" * 70)
        print(f"Full profile saved to: {PROFILE_OUTPUT_PATH.absolute()}")
        print("View interactively with: snakeviz profiling/profile_output.prof")
        print("=" * 70)



if __name__ == "__main__":
    main()
