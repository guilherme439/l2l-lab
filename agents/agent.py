from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Agent(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def choose_action(self, obs: Dict[str, np.ndarray]) -> int:
        """Given an observation dict with 'observation' and 'action_mask', return an action index."""
        ...
