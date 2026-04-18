from __future__ import annotations

from typing import Any

import torch

from agents.agent import Agent
from alphazoo import Explorer, SearchConfig


class MCTSAgent(Agent):

    def __init__(
        self,
        model: torch.nn.Module,
        is_recurrent: bool,
        search_config: SearchConfig,
        obs_space_format: str,
        name: str = "mcts",
    ) -> None:
        self._model = model
        self._is_recurrent = is_recurrent
        self._search_config = search_config
        self._obs_space_format = obs_space_format
        self.name = name

    def choose_action(self, env: Any) -> int:
        return Explorer.select_action_with_mcts_for(
            env=env,
            model=self._model,
            search_config=self._search_config,
            obs_space_format=self._obs_space_format,
            is_recurrent=self._is_recurrent,
        )
