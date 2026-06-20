from typing import Any, override

import torch

from l2l_lab.agents.agent import Agent
from alphazoo import SearchConfig
from alphazoo.utils.mcts import select_action_with_alphazero_mcts


class AlphaZeroMCTSAgent(Agent):

    def __init__(
        self,
        model: torch.nn.Module,
        is_recurrent: bool,
        search_config: SearchConfig,
        obs_space_format: str,
        recurrent_iterations: int = 1,
        name: str = "alphazero_mcts",
    ) -> None:
        self._model = model
        self._is_recurrent = is_recurrent
        self._recurrent_iterations = recurrent_iterations
        self._search_config = search_config
        self._obs_space_format = obs_space_format
        self.name = name

    @override
    def choose_action(self, env: Any) -> int:
        return select_action_with_alphazero_mcts(
            env=env,
            model=self._model,
            search_config=self._search_config,
            obs_space_format=self._obs_space_format,
            is_recurrent=self._is_recurrent,
            recurrent_iterations=self._recurrent_iterations,
        )
