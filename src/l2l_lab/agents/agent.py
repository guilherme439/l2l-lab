from abc import ABC, abstractmethod
from typing import Any

from pettingzoo import AECEnv


class Agent(ABC):

    name: str

    @abstractmethod
    def choose_action(self, env: AECEnv) -> int:
        """Given a PettingZoo env, return an action index."""
        ...
