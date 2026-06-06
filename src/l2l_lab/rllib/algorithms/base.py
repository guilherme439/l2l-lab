from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from dataclasses import asdict

from l2l_lab.configs.training.network import (BaseNetworkConfig, MLPNetConfig,
                                                SNNetConfig)
from l2l_lab.rllib.modules.networks.conv import ConvDualHeadRLModule
from l2l_lab.rllib.modules.networks.mlp import MLPDualHeadRLModule
from l2l_lab.utils.checkpoint import get_algo_checkpoint_path
import logging

logger = logging.getLogger("l2l_lab")

if TYPE_CHECKING:
    from l2l_lab.configs.training.training_config import TrainingConfig


class BaseAlgorithmTrainer(ABC):

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.algo: Any = None

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        pass

    @abstractmethod
    def build_config(self, env_name: str, obs_space_format, obs_space, act_space):
        pass

    @abstractmethod
    def load_from_checkpoint(self, checkpoint_path: Path):
        pass

    @abstractmethod
    def extract_metrics(self, result: dict[str, Any]) -> dict[str, Any]:
        pass

    def update_opponent_policies(self, model_dir: Path, new_checkpoint: int) -> None:
        """Refresh frozen-opponent policy slots after a new checkpoint lands.
        Default is a no-op; algorithms that run multi-policy self-play
        (e.g. PPO with `use_multiple_policies: true`) override this."""
        pass

    @staticmethod
    def get_adapter_class(network: BaseNetworkConfig):
        if isinstance(network, (MLPNetConfig, SNNetConfig)):
            return MLPDualHeadRLModule
        return ConvDualHeadRLModule

    def get_rl_module_spec(self, obs_space, obs_space_format, act_space) -> MultiRLModuleSpec:
        adapter_class = self.get_adapter_class(self.config.network)

        model_config = {
            "network_config": asdict(self.config.network),
        }

        if adapter_class == ConvDualHeadRLModule:
            model_config["obs_space_format"] = obs_space_format

        return MultiRLModuleSpec(
            rl_module_specs={
                "shared_policy": RLModuleSpec(
                    module_class=adapter_class,
                    observation_space=obs_space,
                    action_space=act_space,
                    model_config=model_config,
                ),
            },
        )

    def load_checkpoint_for_continue(
        self,
        rllib_config,
        model_dir: Path,
        target_iteration: Optional[int] = None,
    ) -> int:
        algo_checkpoint_path = get_algo_checkpoint_path(model_dir, target_iteration)
        if algo_checkpoint_path is None or not algo_checkpoint_path.exists():
            logger.info("\nNo existing checkpoint found. Starting fresh training...")
            logger.info(f"\nBuilding {self.algorithm_name.upper()} algorithm...\n")
            self.algo = rllib_config.build_algo()
            logger.info("\n✓ Algorithm built successfully!")
            return 0

        logger.info("\nContinuing training with current config...")
        logger.info(f"Building {self.algorithm_name.upper()} algorithm with new config...\n")
        self.algo = rllib_config.build_algo()
        logger.info("\n✓ Algorithm built successfully!")

        logger.info("Restoring weights from checkpoint...")
        self.algo.restore_from_path(str(algo_checkpoint_path.absolute()))
        logger.info("\n✓ Weights restored from checkpoint")

        start_iteration = int(algo_checkpoint_path.parent.name)

        if target_iteration is not None:
            logger.info(f"✓ Loaded checkpoint from iteration {start_iteration} (requested: {target_iteration})")
        else:
            logger.info(f"✓ Resuming from iteration {start_iteration}")

        return start_iteration
