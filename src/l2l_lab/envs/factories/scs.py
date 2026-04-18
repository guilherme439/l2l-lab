from typing import Any

from pettingzoo import AECEnv


def create_scs_env(**kwargs: Any) -> AECEnv:
    from rl_scs.SCS_Game import SCS_Game
    
    config = kwargs.pop("config", "")
    seed = kwargs.pop("seed", None)
    debug = kwargs.pop("debug", False)
    action_mask_location = kwargs.pop("action_mask_location", "obs")
    obs_space_format = kwargs.pop("obs_space_format", "channels_first")
    
    env = SCS_Game(
        game_config_path=config,
        seed=seed,
        debug=debug,
        action_mask_location=action_mask_location,
        obs_space_format=obs_space_format,
        **kwargs,
    )
    env.simulation_mode = False
    return env
