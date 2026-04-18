from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphazoo import SearchConfig


def load_search_config(path: str) -> "SearchConfig":
    """
    Load an alphazoo SearchConfig from YAML.

    Subtree reuse across consecutive calls is not currently supported on the l2l-lab side — so
    the YAML value is overridden to keep behaviour consistent regardless of what the config says.
    """
    from alphazoo import SearchConfig

    config = SearchConfig.from_yaml(path)
    config.simulation.keep_subtree = False
    return config
