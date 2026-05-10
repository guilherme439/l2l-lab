from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

from l2l_lab.configs.common.EnvConfig import EnvConfig

from ..utils import dataclass_from_dict
from .backends.base import BaseBackendConfig, backend_config_from_dict
from .CommonConfig import CommonConfig
from .EvaluationConfig import EvaluationConfig
from .network import BaseNetworkConfig, network_config_from_dict
from .ReportingConfig import ReportingConfig


@dataclass
class TrainingConfig:
    name: str
    common: CommonConfig
    env: EnvConfig
    network: BaseNetworkConfig
    backend: BaseBackendConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

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

        if "network" not in data:
            raise ValueError("training config is missing required 'network' section")
        network = network_config_from_dict(data["network"])

        if "backend" not in data:
            raise ValueError("training config is missing required 'backend' section")
        backend = backend_config_from_dict(data["backend"])

        evaluation = EvaluationConfig.from_dict(data.get("evaluation", {}))

        reporting = dataclass_from_dict(ReportingConfig, data.get("reporting", {}))

        return cls(
            name=data["name"],
            common=common,
            env=env,
            network=network,
            backend=backend,
            evaluation=evaluation,
            reporting=reporting,
        )
