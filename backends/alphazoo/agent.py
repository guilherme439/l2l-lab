from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from agents.agent import Agent


class AlphaZooPolicyAgent(Agent):

    def __init__(
        self,
        network_manager: Any,
        obs_to_state: Callable[[Any, Any], torch.Tensor],
        label: str = "alphazoo",
    ):
        self._network_manager = network_manager
        self._obs_to_state = obs_to_state
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    def choose_action(self, obs: Dict[str, np.ndarray]) -> int:
        state = self._obs_to_state(obs, None)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.float32)

        self._network_manager.get_model().eval()
        with torch.no_grad():
            policy_logits, _ = self._network_manager.inference(state, training=False, iters_to_do=1)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        return action
