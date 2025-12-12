from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from ..PolicyConfig import PolicyConfig


@dataclass
class AlgoPPOConfig:
    use_curiosity: bool = False
    curiosity_coeff: float = 0.05
    
    num_env_runners: int = 0
    rollout_fragment_length: int = 64
    train_batch_size_per_learner: int = 512
    minibatch_size: int = 64
    num_epochs: int = 3
    lr: Union[float, List[Tuple[int, float]]] = 0.0001
    gamma: float = 0.99
    entropy_coeff: float = 0.05
    vf_loss_coeff: float = 0.5
    clip_param: float = 0.2
    use_kl_loss: bool = True
    kl_coeff: float = 0.2
    kl_target: float = 0.01
    
    policy: Optional[PolicyConfig] = None
    
