from dataclasses import dataclass


@dataclass
class AlgoIMPALAConfig:
    num_env_runners: int = 2
    rollout_fragment_length: int = 50
    num_learners: int = 1
    lr: float = 0.0005
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    vf_loss_coeff: float = 0.5
    grad_clip: float = 40.0
    vtrace: bool = True
    vtrace_clip_rho_threshold: float = 1.0
    vtrace_clip_pg_rho_threshold: float = 1.0
