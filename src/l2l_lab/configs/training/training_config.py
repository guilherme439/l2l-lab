from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import TypeAdapter

from l2l_lab.configs.common.env_config import EnvConfig

from .backends import BackendConfig
from .common_config import CommonConfig
from .evaluation_config import EvaluationConfig
from .network import NetworkConfig
from .reporting_config import ReportingConfig


@dataclass
class TrainingConfig:
    name: str
    common: CommonConfig
    env: EnvConfig
    network: NetworkConfig
    backend: BackendConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        return cls.from_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        return TypeAdapter(cls).validate_python(data)
