from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib.networks.adapters.conv import ConvDualHeadRLModule
from rllib.networks.adapters.mlp import MLPDualHeadRLModule
from checkpoint_utils import CheckpointData, load_checkpoint_data, trim_metrics_to_iteration, get_algo_checkpoint_path

if TYPE_CHECKING:
    from Trainer import Trainer


class BaseAlgorithmTrainer(ABC):
    
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.config = trainer.config
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
    def extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def get_rl_module_spec(self, obs_space, obs_space_format, act_space) -> MultiRLModuleSpec:
        network_class = self.config.network.get_network_class()
        adapter_class = self.config.network.get_adapter_class()
        
        model_config = {
            "network_class": network_class,
            "network_kwargs": self.config.network.to_kwargs(),
            "architecture": self.config.network.architecture,
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
    ) -> Tuple[int, Optional[CheckpointData]]:
        algo_checkpoint_path = get_algo_checkpoint_path(model_dir, target_iteration)
        if algo_checkpoint_path is None or not algo_checkpoint_path.exists():
            print("\nNo existing checkpoint found. Starting fresh training...")
            print(f"\nBuilding {self.algorithm_name.upper()} algorithm...\n")
            self.algo = rllib_config.build_algo()
            print("\n✓ Algorithm built successfully!")
            return 0, None
        
        print("\nContinuing training with current config...")
        print(f"Building {self.algorithm_name.upper()} algorithm with new config...\n")
        self.algo = rllib_config.build_algo()
        print("\n✓ Algorithm built successfully!")
        
        print("Restoring weights from checkpoint...")
        self.algo.restore_from_path(str(algo_checkpoint_path.absolute()))
        print("✓ Weights restored from checkpoint")
        
        cp_data = load_checkpoint_data(model_dir, iteration=target_iteration)
        start_iteration = cp_data.iteration if cp_data else 0
        
        if target_iteration is not None and cp_data and cp_data.metrics:
            cp_data.metrics = trim_metrics_to_iteration(cp_data.metrics, start_iteration)
            print(f"✓ Loaded checkpoint from iteration {start_iteration} (requested: {target_iteration})")
        else:
            print(f"✓ Resuming from iteration {start_iteration}")
        
        return start_iteration, cp_data
    
