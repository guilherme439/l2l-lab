from dataclasses import dataclass


@dataclass
class PPOConfig:

    num_env_runners: int = 0
    rollout_fragment_length: int = 64

    train_batch_size_per_learner: int = 256
    minibatch_size: int = 16

    lr: float = 5e-4
    gamma: float = 0.995
    entropy_coeff: float = 0.0
