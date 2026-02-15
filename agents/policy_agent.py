from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from agents.agent import Agent


class PolicyAgent(Agent):

    def __init__(self, backbone: torch.nn.Module, obs_space_format: str = "channels_first", label: str = "policy"):
        self._backbone = backbone
        self._obs_space_format = obs_space_format
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    def choose_action(self, obs: Dict[str, np.ndarray]) -> int:
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        action_mask = obs["action_mask"]

        if self._obs_space_format == "channels_last":
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        with torch.no_grad():
            policy_logits, _ = self._backbone(obs_tensor)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        return action
