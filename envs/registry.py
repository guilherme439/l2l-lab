from typing import Any, Callable, Dict

from pettingzoo import AECEnv


EnvFactory = Callable[..., AECEnv]

ENV_REGISTRY: Dict[str, EnvFactory] = {}


def _register_builtin_envs() -> None:
    from .factories.scs import create_scs_env
    from .factories.pettingzoo_classic import create_tictactoe_env
    
    ENV_REGISTRY["scs"] = create_scs_env
    ENV_REGISTRY["tictactoe"] = create_tictactoe_env


def create_env(name: str, **kwargs: Any) -> AECEnv:
    if not ENV_REGISTRY:
        _register_builtin_envs()
    
    if name not in ENV_REGISTRY:
        available = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Unknown environment: {name}. Available: {available}")
    
    factory = ENV_REGISTRY[name]
    return factory(**kwargs)


def register_env(name: str, factory: EnvFactory) -> None:
    ENV_REGISTRY[name] = factory


def list_envs() -> list[str]:
    if not ENV_REGISTRY:
        _register_builtin_envs()
    return list(ENV_REGISTRY.keys())
