from typing import Annotated, Union

from pydantic import Field

from .agent_config import BaseAgentConfig
from .alphazero_mcts_agent_config import AlphaZeroMCTSAgentConfig
from .policy_agent_config import PolicyAgentConfig
from .random_agent_config import RandomAgentConfig
from .traditional_mcts_agent_config import TraditionalMCTSAgentConfig

AgentConfig = Annotated[
    Union[
        PolicyAgentConfig,
        RandomAgentConfig,
        AlphaZeroMCTSAgentConfig,
        TraditionalMCTSAgentConfig,
    ],
    Field(discriminator="agent_type"),
]

__all__ = [
    "BaseAgentConfig",
    "PolicyAgentConfig",
    "RandomAgentConfig",
    "AlphaZeroMCTSAgentConfig",
    "TraditionalMCTSAgentConfig",
    "AgentConfig",
]
