from typing import Any

from pettingzoo import AECEnv


def create_tictactoe_env(**kwargs: Any) -> AECEnv:
    from pettingzoo.classic import tictactoe_v3
    return tictactoe_v3.env(**kwargs)
