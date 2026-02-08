from typing import Any

from pettingzoo import AECEnv


def create_tictactoe_env(**kwargs: Any) -> AECEnv:
    from pettingzoo.classic import tictactoe_v3
    return tictactoe_v3.env(**kwargs)


def create_leduc_holdem_env(**kwargs: Any) -> AECEnv:
    from pettingzoo.classic import leduc_holdem_v4
    return leduc_holdem_v4.env(**kwargs)


def create_connect_four_env(**kwargs: Any) -> AECEnv:
    from pettingzoo.classic import connect_four_v3
    return connect_four_v3.env(**kwargs)
