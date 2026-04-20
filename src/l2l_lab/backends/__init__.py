from __future__ import annotations

from typing import Dict, Type

from l2l_lab.backends.backend_base import AlgorithmBackend

BACKEND_REGISTRY: Dict[str, Type[AlgorithmBackend]] = {}

_builtins_registered = False


def _register_builtins() -> None:
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True

    from l2l_lab.backends.rllib.backend import RLlibBackend
    register_backend("rllib", RLlibBackend)

    try:
        import alphazoo  # noqa: F401

        from l2l_lab.backends.alphazoo.backend import AlphaZooBackend
        register_backend("alphazoo", AlphaZooBackend)
    except ImportError:
        pass


def register_backend(name: str, cls: Type[AlgorithmBackend]) -> None:
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> Type[AlgorithmBackend]:
    _register_builtins()
    if name not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")
    return BACKEND_REGISTRY[name]
