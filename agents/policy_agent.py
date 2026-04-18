from __future__ import annotations

from typing import Any, Callable

import torch

from agents.agent import Agent
from backends.obs_utils import obs_to_state_provider


class PolicyAgent(Agent):

    def __init__(
        self,
        model: torch.nn.Module,
        obs_space_format: str,
        name: str = "policy",
    ) -> None:
        self._model = model
        self._obs_to_state: Callable[[Any, Any], torch.Tensor] = obs_to_state_provider(obs_space_format)
        self.name = name

    def choose_action(self, env: Any) -> int:
        agent_id = env.agent_selection
        obs = env.observe(agent_id)
        action_mask = obs["action_mask"]
        state = self._obs_to_state(obs, agent_id)

        with torch.no_grad():
            policy_logits, _ = self._model(state)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
