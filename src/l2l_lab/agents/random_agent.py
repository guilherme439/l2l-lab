from __future__ import annotations

import random
from typing import Any

import numpy as np

from l2l_lab.agents.agent import Agent


class RandomAgent(Agent):

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def choose_action(self, env: Any) -> int:
        obs = env.observe(env.agent_selection)
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        return int(random.choice(valid_actions))
