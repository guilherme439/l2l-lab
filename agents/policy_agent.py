from __future__ import annotations

import numpy as np
import torch

from agents.agent import Agent


class PolicyAgent(Agent):

    def __init__(self, model: torch.nn.Module, name: str = "policy"):
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choose_action(self, state: torch.Tensor, action_mask: np.ndarray) -> int:
        with torch.no_grad():
            policy_logits, _ = self._model(state)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
