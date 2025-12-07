from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rllib.DualHeadRLModule import DualHeadRLModule

if TYPE_CHECKING:
    from rllib.Trainer import Trainer


class BaseAlgorithmTrainer(ABC):
    
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.algo = None
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        pass
    
    @abstractmethod
    def build_config(self, obs_space, act_space):
        pass
    
    @abstractmethod
    def load_from_checkpoint(self, checkpoint_path: Path):
        pass
    
    @abstractmethod
    def extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def get_rl_module_spec(self, obs_space, act_space) -> MultiRLModuleSpec:
        network_class = self.config.network.get_network_class()
        return MultiRLModuleSpec(
            rl_module_specs={
                "shared_policy": RLModuleSpec(
                    module_class=DualHeadRLModule,
                    observation_space=obs_space,
                    action_space=act_space,
                    model_config={
                        "network_class": network_class,
                        "network_kwargs": self.config.network.to_kwargs(),
                    },
                ),
            },
        )
    
    def load_checkpoint_for_continue(self, rllib_config, model_dir: Path) -> int:
        algo_checkpoint_path = model_dir / "algo_checkpoint"
        if not algo_checkpoint_path.exists():
            print("\nNo existing checkpoint found. Starting fresh training...")
            print(f"\nBuilding {self.algorithm_name.upper()} algorithm...\n")
            self.algo = rllib_config.build_algo()
            print("\n✓ Algorithm built successfully!")
            return 0
        
        print("\nLoading algorithm from checkpoint for continued training...")
        self.algo = self.load_from_checkpoint(algo_checkpoint_path)
        
        start_iteration = 0
        latest_cp = self._get_latest_checkpoint(model_dir)
        if latest_cp:
            cp_data = torch.load(latest_cp, weights_only=False)
            start_iteration = cp_data.get("iteration", 0)
        else:
            model_cp = model_dir / "model.cp"
            if model_cp.exists():
                cp_data = torch.load(model_cp, weights_only=False)
                start_iteration = cp_data.get("iteration", 0)
        
        print(f"✓ Resuming from iteration {start_iteration}")
        return start_iteration
    
    def _get_latest_checkpoint(self, model_dir: Path):
        checkpoints_dir = model_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None
        
        checkpoints = list(checkpoints_dir.glob("model_iter_*.cp"))
        if not checkpoints:
            return None
        
        def get_iter(p: Path) -> int:
            return int(p.stem.split("_")[-1])
        
        return max(checkpoints, key=get_iter)
