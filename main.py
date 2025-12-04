import warnings

import ray

from rllib.SCSTrainer import SCSTrainer

warnings.filterwarnings("ignore", category=DeprecationWarning)


CONFIG_PATH = "configs/files/training_config.yml"

def main():
    ray.init(ignore_reinit_error=True)
    
    try:
        trainer = SCSTrainer(CONFIG_PATH)
        trainer.train()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ray.shutdown()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
