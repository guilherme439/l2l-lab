from typing import Any, Callable, override

from pettingzoo import AECEnv
import torch

from l2l_lab.agents.agent import Agent
from l2l_lab.backends.obs_utils import obs_to_state_provider


class PolicyAgent(Agent):

    def __init__(
        self,
        model: torch.nn.Module,
        obs_space_format: str,
        is_recurrent: bool = False,
        recurrent_iterations: int = 1,
        name: str = "policy",
    ) -> None:
        self._model = model
        self._obs_to_state: Callable[[Any, Any], torch.Tensor] = obs_to_state_provider(obs_space_format)
        self._is_recurrent = is_recurrent
        self._recurrent_iterations = recurrent_iterations
        self.name = name

    @override
    def choose_action(self, env: AECEnv) -> int:
        agent_id = env.agent_selection
        obs = env.observe(agent_id)
        action_mask = obs["action_mask"]
        state = self._obs_to_state(obs, agent_id)

        with torch.no_grad():
            if self._is_recurrent:
                (policy_logits, _), _ = self._model(state, iters_to_do=self._recurrent_iterations)
            else:
                policy_logits, _ = self._model(state)
            policy_logits = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)
            policy_logits[action_mask == 0] = float("-inf")
            probs = torch.softmax(policy_logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
