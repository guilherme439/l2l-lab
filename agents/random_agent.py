from __future__ import annotations

import random
from typing import Dict

import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):

    @property
    def name(self) -> str:
        return "random"

    def choose_action(self, obs: Dict[str, np.ndarray]) -> int:
        valid_actions = np.where(obs["action_mask"] == 1)[0]
        return random.choice(valid_actions)
