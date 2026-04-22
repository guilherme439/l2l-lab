from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

from l2l_lab.configs.common.EnvConfig import EnvConfig

from ..utils import dataclass_from_dict
from .backends.base import BaseBackendConfig, backend_config_from_dict
from .CommonConfig import CommonConfig
from .EvaluationConfig import EvaluationConfig
from .NetworkConfig import NetworkConfig


@dataclass
class TrainingConfig:
    name: str
    common: CommonConfig
    env: EnvConfig
    network: NetworkConfig
    backend: BaseBackendConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "common" not in data:
            raise ValueError("training config is missing required 'common' section")
        common = dataclass_from_dict(CommonConfig, data["common"])

        env_data = data.get("env", {})
        env = EnvConfig(
            name=env_data.get("name"),
            obs_space_format=env_data.get("obs_space_format", "channels_first"),
            kwargs=env_data.get("kwargs", {}),
        )

        network_data = dict(data.get("network", {}))
        network = NetworkConfig(
            architecture=network_data.pop("architecture"),
            kwargs=network_data,
        )

        if "backend" not in data:
            raise ValueError("training config is missing required 'backend' section")
        backend = backend_config_from_dict(data["backend"])

        evaluation = EvaluationConfig.from_dict(data.get("evaluation", {}))

        return cls(
            name=data["name"],
            common=common,
            env=env,
            network=network,
            backend=backend,
            evaluation=evaluation,
        )
