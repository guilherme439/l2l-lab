from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from l2l_lab.backends.obs_utils import obs_to_state_provider
from l2l_lab.configs.common.EnvConfig import EnvConfig

from .types import ProbeState


def run_probe_states(
    model: Optional[torch.nn.Module],
    env_config: EnvConfig,
    probe_states: list[ProbeState],
) -> list[dict[str, Any]]:
    """Feed each probe observation through ``model`` and collect its output.

    Returns one dict per probe with: label, description, current_player,
    legal_actions, policy (per-action probabilities, masked), value, and
    logit summary stats. Illegal-action probabilities are reported as 0.0.
    """
    if model is None or not probe_states:
        return []

    obs_to_state = obs_to_state_provider(env_config.obs_space_format)
    was_training = model.training
    model.eval()
    results: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            for probe in probe_states:
                action_mask = np.asarray(probe.observation["action_mask"])
                state = obs_to_state(probe.observation, probe.current_player)

                policy_logits, value = model(state)
                logits_flat = policy_logits.reshape(policy_logits.shape[0], -1).squeeze(0)

                logits_np = logits_flat.detach().cpu().numpy().astype(np.float64)
                masked_logits = logits_flat.clone()
                mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=masked_logits.device)
                masked_logits[~mask_tensor] = float("-inf")
                probs = torch.softmax(masked_logits, dim=-1).detach().cpu().numpy()

                value_scalar = float(value.detach().cpu().numpy().ravel()[0]) if value is not None else float("nan")
                legal_actions = [int(i) for i, m in enumerate(action_mask) if m]

                results.append({
                    "label": probe.label,
                    "description": probe.description,
                    "current_player": probe.current_player,
                    "legal_actions": legal_actions,
                    "policy": [float(p) for p in probs.tolist()],
                    "value": value_scalar,
                    "logits_min": float(np.min(logits_np)),
                    "logits_max": float(np.max(logits_np)),
                    "logits_mean": float(np.mean(logits_np)),
                    "logits_std": float(np.std(logits_np)),
                })
    finally:
        if was_training:
            model.train()

    return results
