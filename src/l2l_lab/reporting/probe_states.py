from __future__ import annotations

from typing import Callable

import numpy as np

from .types import ProbeState

ProbeStateProvider = Callable[[], list[ProbeState]]

PROBE_STATES_REGISTRY: dict[str, ProbeStateProvider] = {}


def register_probe_states(env_name: str, provider: ProbeStateProvider) -> None:
    PROBE_STATES_REGISTRY[env_name] = provider


def get_probe_states(env_name: str) -> list[ProbeState]:
    if not PROBE_STATES_REGISTRY:
        _register_builtin_providers()
    provider = PROBE_STATES_REGISTRY.get(env_name)
    if provider is None:
        return []
    return provider()


def _register_builtin_providers() -> None:
    register_probe_states("connect_four", _connect_four_probe_states)


def _cf_board(placements: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """Build a Connect Four observation dict from a placement list.

    ``placements`` is a list of ``(row, col, player)`` where ``player`` is 0
    for the current player (channel 0) and 1 for the opponent (channel 1).
    Row 0 is the top of the 6x7 board. ``action_mask`` is derived from column
    occupancy: a column is legal iff its top cell is empty.
    """
    board = np.zeros((6, 7, 2), dtype=np.int8)
    for row, col, player in placements:
        board[row, col, player] = 1
    action_mask = np.ones(7, dtype=np.int8)
    for col in range(7):
        if board[0, col, 0] != 0 or board[0, col, 1] != 0:
            action_mask[col] = 0
    return {"observation": board, "action_mask": action_mask}


def _connect_four_probe_states() -> list[ProbeState]:
    probes: list[ProbeState] = []

    probes.append(ProbeState(
        label="empty_board",
        observation=_cf_board([]),
        current_player="player_0",
        description="Opening move. A well-trained policy should heavily prefer the center column (3).",
    ))

    probes.append(ProbeState(
        label="center_opening_response",
        observation=_cf_board([(5, 3, 1)]),
        current_player="player_0",
        description="Opponent opened center. Best response is to also play center (column 3) to stack.",
    ))

    probes.append(ProbeState(
        label="immediate_win_available",
        observation=_cf_board([
            (5, 2, 0), (5, 3, 0), (5, 4, 0),
            (4, 2, 1), (4, 3, 1),
        ]),
        current_player="player_0",
        description="Three-in-a-row on bottom row (cols 2-4). Column 1 or 5 wins immediately.",
    ))

    probes.append(ProbeState(
        label="must_block_threat",
        observation=_cf_board([
            (5, 1, 1), (5, 2, 1), (5, 3, 1),
            (4, 2, 0), (4, 3, 0),
        ]),
        current_player="player_0",
        description="Opponent threatens four-in-a-row on bottom (cols 1-3). Must block column 0 or 4.",
    ))

    probes.append(ProbeState(
        label="diagonal_fork",
        observation=_cf_board([
            (5, 3, 0), (4, 4, 0),
            (5, 4, 1), (5, 5, 1),
        ]),
        current_player="player_0",
        description="Building a diagonal. Playing column 5 creates a double threat.",
    ))

    return probes
