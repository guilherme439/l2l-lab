from __future__ import annotations

import random

import numpy as np
import torch

from agents.agent import Agent


class RandomAgent(Agent):

    @property
    def name(self) -> str:
        return "random"

    def choose_action(self, state: torch.Tensor, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask == 1)[0]
        return random.choice(valid_actions)
