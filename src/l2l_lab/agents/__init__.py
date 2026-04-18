from l2l_lab.agents.agent import Agent
from l2l_lab.agents.policy_agent import PolicyAgent
from l2l_lab.agents.random_agent import RandomAgent

__all__ = ["Agent", "PolicyAgent", "RandomAgent"]

try:
    import alphazoo  # noqa: F401

    from l2l_lab.agents.mcts_agent import MCTSAgent

    __all__.append("MCTSAgent")
except ImportError:
    pass
