import ray

from rllib.train_scs_rllib import SCSTrainer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    ray.init(ignore_reinit_error=True, num_cpus=4)
    
    try:
        trainer = SCSTrainer(debug=False)
        trainer.train_ppo(num_iterations=500, model_name="test")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        ray.shutdown()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
