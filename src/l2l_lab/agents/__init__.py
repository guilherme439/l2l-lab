from l2l_lab.agents.agent import Agent
from l2l_lab.agents.policy_agent import PolicyAgent
from l2l_lab.agents.random_agent import RandomAgent

__all__ = ["Agent", "PolicyAgent", "RandomAgent"]

try:
    import alphazoo  # noqa: F401

    from l2l_lab.agents.alphazero_mcts_agent import AlphaZeroMCTSAgent
    from l2l_lab.agents.traditional_mcts_agent import TraditionalMCTSAgent

    __all__.extend(["AlphaZeroMCTSAgent", "TraditionalMCTSAgent"])
except ImportError:
    pass
