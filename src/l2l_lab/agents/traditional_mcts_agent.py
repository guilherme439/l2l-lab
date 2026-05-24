from __future__ import annotations

from typing import Any

from l2l_lab.agents.agent import Agent
from alphazoo import SearchConfig
from alphazoo.utils import select_action_with_traditional_mcts


class TraditionalMCTSAgent(Agent):

    def __init__(
        self,
        search_config: SearchConfig,
        obs_space_format: str,
        name: str = "traditional_mcts",
    ) -> None:
        self._search_config = search_config
        self._obs_space_format = obs_space_format
        self.name = name

    def choose_action(self, env: Any) -> int:
        return select_action_with_traditional_mcts(
            env=env,
            search_config=self._search_config,
            obs_space_format=self._obs_space_format,
        )
