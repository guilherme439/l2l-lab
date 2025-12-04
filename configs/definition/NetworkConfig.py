from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class NetworkConfig:
    architecture: str = "ResNet"
    num_filters: int = 64
    num_blocks: int = 3
    batch_norm: bool = False
    hex: bool = False
    
    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "num_filters": self.num_filters,
            "num_blocks": self.num_blocks,
            "batch_norm": self.batch_norm,
            "hex": self.hex,
        }
