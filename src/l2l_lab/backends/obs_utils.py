from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def _hwc_to_chw(observation: np.ndarray) -> torch.Tensor:
    """Convert a HWC numpy observation to a contiguous CHW float32 tensor with batch dim."""
    return torch.from_numpy(
        np.ascontiguousarray(observation.transpose(2, 0, 1), dtype=np.float32)
    ).unsqueeze(0)


def obs_to_state_provider(obs_space_format: str) -> Callable[[Any, Any], torch.Tensor]:
    """Return an obs-to-tensor conversion function for the given format."""
    if obs_space_format == "channels_last":
        def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
            return _hwc_to_chw(obs["observation"])
        return obs_to_state

    def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
        return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
    return obs_to_state


def make_wrapper(env: Any, obs_space_format: str = "channels_last"):
    """
    Create a PettingZooWrapper configured for the given obs format.

    The wrapper's transpose behavior is controlled by ``observation_format``
    and ``network_input_format``. l2l-lab networks always expect channels_first.
    """
    from alphazoo import PettingZooWrapper

    return PettingZooWrapper(
        env,
        observation_format=obs_space_format,
        network_input_format="channels_first",
    )
