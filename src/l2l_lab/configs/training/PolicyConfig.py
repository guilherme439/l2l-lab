from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PolicyConfig:
    use_multiple_policies: bool = False
    number_previous_policies: int = 0
    main_policy_ratio: float = 1.0
    random_policy_ratio: float = 0.0
    
    def __post_init__(self):
        if self.use_multiple_policies:
            total = self.main_policy_ratio + self.random_policy_ratio
            if self.number_previous_policies == 0 and abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"With number_previous_policies=0, main_policy_ratio + random_policy_ratio must equal 1.0, "
                    f"got {total}"
                )
            if total > 1.0 + 1e-6:
                raise ValueError(
                    f"main_policy_ratio + random_policy_ratio cannot exceed 1.0, got {total}"
                )
    
    @property
    def previous_policy_ratio(self) -> float:
        return 1.0 - self.main_policy_ratio - self.random_policy_ratio
    
    def get_policy_weights(self, num_available_checkpoints: int) -> Dict[str, float]:
        prev_ratio = self.previous_policy_ratio
        
        if self.number_previous_policies > 0 and num_available_checkpoints > 0:
            n_active = min(self.number_previous_policies, num_available_checkpoints)
            per_policy_ratio = prev_ratio / n_active
            unused_ratio = prev_ratio - (per_policy_ratio * n_active)
            
            weights = {"main_policy": self.main_policy_ratio + unused_ratio}
            
            for i in range(n_active):
                weights[f"checkpoint_{i}"] = per_policy_ratio
        else:
            weights = {"main_policy": self.main_policy_ratio + prev_ratio}
        
        if self.random_policy_ratio > 0:
            weights["random_policy"] = self.random_policy_ratio
        
        return weights
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyConfig":
        return cls(
            use_multiple_policies=data.get("use_multiple_policies", False),
            number_previous_policies=data.get("number_previous_policies", 0),
            main_policy_ratio=data.get("main_policy_ratio", 1.0),
            random_policy_ratio=data.get("random_policy_ratio", 0.0),
        )
