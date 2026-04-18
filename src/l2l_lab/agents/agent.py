from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):

    name: str

    @abstractmethod
    def choose_action(self, env: Any) -> int:
        """Given a PettingZoo env, return an action index."""
        ...
