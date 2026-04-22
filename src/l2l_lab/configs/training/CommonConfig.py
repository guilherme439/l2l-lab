from dataclasses import dataclass


@dataclass
class CommonConfig:
    plot_interval: int = 0
    info_interval: int = 0
    eval_graph_split: int = 0
    checkpoint_interval: int = 0
