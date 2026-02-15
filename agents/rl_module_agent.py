from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from agents.agent import Agent


class RLModuleAgent(Agent):

    def __init__(self, rl_module, label: str = "rl_module"):
        self._rl_module = rl_module
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    def choose_action(self, obs: Dict[str, np.ndarray]) -> int:
        obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        action_mask = obs["action_mask"]
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            batch = {"obs": {"observation": obs_tensor, "action_mask": mask_tensor}}
            output = self._rl_module.forward_inference(batch)
            logits = output["action_dist_inputs"].squeeze(0)
            logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        return action
