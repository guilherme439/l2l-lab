from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphazoo import SearchConfig


def load_search_config(path: str) -> "SearchConfig":
    """
    Load an alphazoo SearchConfig from YAML for inference-time MCTS.

    Training-only knobs — subtree reuse and root exploration noise — are zeroed out regardless
    of what the YAML specifies, so the returned config is always safe to feed into a
    non-training ``Explorer``.
    """
    from alphazoo import SearchConfig

    config = SearchConfig.from_yaml(path)
    config.simulation.keep_subtree = False
    config.exploration.root_exploration_fraction = 0.0
    config.exploration.root_dist_alpha = 0.0
    return config
