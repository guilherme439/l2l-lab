from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Agent(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def choose_action(self, state: torch.Tensor, action_mask: np.ndarray) -> int:
        """Given a state tensor and action mask, return an action index."""
        ...
