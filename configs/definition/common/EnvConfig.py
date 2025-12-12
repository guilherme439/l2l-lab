from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class EnvConfig:
    name: str
    obs_space_format: Literal["channels_first", "channels_last", "flat"] = "channels_first"
    kwargs: Dict[str, Any] = field(default_factory=dict)
