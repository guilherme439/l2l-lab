import argparse
import sys
import traceback
import warnings
from pathlib import Path

from l2l_lab.testing.tester import Tester
from l2l_lab.training.trainer import Trainer
import logging

logger = logging.getLogger("l2l_lab")

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

    try:
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
        logger.info("\nDone!")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def train(config_path: str):
    trainer = Trainer(config_path)
    trainer.train()


def test(config_path: str):
    try:
        tester = Tester(config_path)
        tester.test()

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
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

        logger.info("\n" + "=" * 70)
        logger.info("PROFILING SUMMARY (top 20 by total time, all threads)")
        logger.info("=" * 70)
        stats.sort("ttot", "desc")
        header = f"{'name':<50} {'ncall':>8} {'ttot':>8} {'tsub':>8} {'tavg':>8}"
        logger.info(header)
        for stat in stats[:20]:
            logger.info(f"{stat.full_name[:50]:<50} {stat.ncall:>8} {stat.ttot:>8.4f} {stat.tsub:>8.4f} {stat.tavg:>8.4f}")

        logger.info("=" * 70)
        logger.info(f"Full profile saved to: {PROFILE_OUTPUT_PATH.absolute()}")
        logger.info("View interactively with: snakeviz profiling/profile_output.prof")
        logger.info("=" * 70)



if __name__ == "__main__":
    main()
