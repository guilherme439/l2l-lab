from __future__ import annotations

from typing import Any, Callable

import torch


def make_obs_to_state(obs_space_format: str) -> Callable[[Any, Any], torch.Tensor]:
    """Return an obs-to-tensor conversion function for the given format."""
    if obs_space_format == "channels_last":
        def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
            t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
            return t.permute(0, 3, 1, 2)
        return obs_to_state

    def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
        return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
    return obs_to_state


def make_wrapper(env: Any, architecture: str, obs_space_format: str):
    """
    Create a PettingZooWrapper appropriate for the given architecture and obs format.

    For channels-first (default) environments returns a plain PettingZooWrapper.
    For channels-last environments returns a subclass that permutes the axes.
    """
    from alphazoo import PettingZooWrapper

    if obs_space_format == "channels_last":
        class ChannelsLastWrapper(PettingZooWrapper):
            def obs_to_state(self, obs: Any, agent_id: Any) -> torch.Tensor:
                t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
                return t.permute(0, 3, 1, 2)

        return ChannelsLastWrapper(env)

    return PettingZooWrapper(env)
