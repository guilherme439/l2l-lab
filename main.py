import argparse
import cProfile
import pstats
from pathlib import Path

import ray

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


TRAINING_CONFIG_PATH = "configs/files/training/hex_training_config.yml"
TESTING_CONFIG_PATH = "configs/files/testing/testing_config.yml"
PROFILE_OUTPUT_PATH = Path("profiling/profile_output.prof")


def train(config_path: str):
    from rllib.SCSTrainer import SCSTrainer
    
    ray.init(ignore_reinit_error=True)
    
    try:
        trainer = SCSTrainer(config_path)
        trainer.train()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ray.shutdown()


def test(config_path: str):
    from rllib.SCSTester import SCSTester
    
    try:
        tester = SCSTester(config_path)
        tester.test()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def run_with_profiling(func, *args):
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        func(*args)
    finally:
        profiler.disable()
        profiler.dump_stats(str(PROFILE_OUTPUT_PATH))
        
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY (top 30 by cumulative time)")
        print("=" * 70)
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        
        print("=" * 70)
        print(f"Full profile saved to: {PROFILE_OUTPUT_PATH.absolute()}")
        print("View interactively with: snakeviz profile_output.prof")
        print("=" * 70)


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
        help="Enable cProfile profiling and save results to profile_output.prof"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        config_path = args.config or TRAINING_CONFIG_PATH
        if args.profile:
            run_with_profiling(train, config_path)
        else:
            train(config_path)
    elif args.mode == "test":
        config_path = args.config or TESTING_CONFIG_PATH
        if args.profile:
            run_with_profiling(test, config_path)
        else:
            test(config_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
