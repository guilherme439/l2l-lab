import argparse
import os
import shutil
import signal
import subprocess
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
PROFILE_OUTPUT_PATH = Path("profiling/profile.speedscope.json")


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
        help="Profile the run with py-spy (main process, all threads) and write a speedscope file"
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
    executable = shutil.which("py-spy")
    if executable is None:
        raise RuntimeError("py-spy not found on PATH. Install with `pip install py-spy`.")

    PROFILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = PROFILE_OUTPUT_PATH.parent / "py-spy.log"

    cmd = [
        executable, "record",
        "--pid", str(os.getpid()),
        "--threads",
        "--rate", "100",
        "--format", "speedscope",
        "--output", str(PROFILE_OUTPUT_PATH),
    ]
    with open(stderr_path, "w") as stderr_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)

    try:
        rc = proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass 
    else:
        stderr_content = stderr_path.read_text().strip()
        raise RuntimeError(
            f"py-spy exited immediately (code {rc}).\n"
            f"stderr:\n{stderr_content}\n\n"
            "On Linux, py-spy needs ptrace permissions:\n"
            "  sudo setcap cap_sys_ptrace=eip $(readlink -f $(which py-spy))\n"
            "  echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope"
        )

    try:
        func(*args)
    finally:
        proc.send_signal(signal.SIGINT)  # SIGINT makes py-spy flush the speedscope file to disk
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()

        logger.info("\n" + "=" * 70)
        logger.info(f"Profile saved to: {PROFILE_OUTPUT_PATH.absolute()}")
        logger.info("Open it at https://www.speedscope.app/")
        logger.info("=" * 70)



if __name__ == "__main__":
    main()
