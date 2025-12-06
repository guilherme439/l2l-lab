from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AlgorithmConfig:
    name: str
    iterations: int
    config: Dict[str, Any] = field(default_factory=dict)
