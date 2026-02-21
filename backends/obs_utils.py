from __future__ import annotations

from typing import Any, Callable

import torch


def make_obs_to_state(architecture: str, obs_space_format: str) -> Callable[[Any, Any], torch.Tensor]:
    """Build an observation-to-state function based on architecture and obs format.

    Conv architectures (ResNet, ConvNet): extract obs["observation"],
    convert to float tensor, permute if channels_last.

    MLP architectures (MLPNet): extract obs["observation"], convert to float tensor.
    """
    is_conv = architecture in ("ResNet", "ConvNet")

    if is_conv and obs_space_format == "channels_last":
        def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
            t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
            return t.permute(0, 3, 1, 2)
    elif is_conv:
        def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
            return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
    else:
        def obs_to_state(obs: Any, agent_id: Any) -> torch.Tensor:
            return torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)

    return obs_to_state


def default_action_mask_fn(env) -> Any:
    """Extract action mask from a PettingZoo env's current agent observation."""
    import numpy as np
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs.get("action_mask", None)
    if mask is not None:
        return np.array(mask, dtype=np.float32)
    return np.ones(env.action_space(agent).n, dtype=np.float32)
