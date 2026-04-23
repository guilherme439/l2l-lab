from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProbeState:
    """A fixed, canonical observation used to probe the policy each snapshot.

    Providers supply pre-built observation dicts (same shape the env produces)
    so probes are robust to non-deterministic envs and decoupled from replay
    semantics.
    """
    label: str
    observation: dict[str, Any]
    current_player: Optional[str] = None
    description: Optional[str] = None


@dataclass
class GameReport:
    """A single captured sample game.

    ``moves`` is a list of ``(agent_id, action, obs_before_action)`` triples;
    carrying the raw pre-action observation keeps the report useful even when
    the env is stochastic.
    """
    p0_name: str
    p1_name: str
    result_from_p0: int
    num_moves: int
    moves: list[tuple[str, Optional[int], dict[str, Any]]] = field(default_factory=list)
