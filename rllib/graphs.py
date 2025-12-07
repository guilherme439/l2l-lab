from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt






def plot_episode_length(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values, "g-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Episode Length Mean")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "episode_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wld_stacked(
    graphs_dir: Path,
    iterations: List[int],
    wins: List[Optional[int]],
    losses: List[Optional[int]],
    draws: List[Optional[int]],
    title: str,
    filename: str,
) -> None:
    valid_data = [
        (i, w, l, d) for i, w, l, d in zip(iterations, wins, losses, draws)
        if w is not None and l is not None and d is not None
    ]
    if not valid_data:
        return
    
    iters = [x[0] for x in valid_data]
    w_vals = [x[1] for x in valid_data]
    l_vals = [x[2] for x in valid_data]
    d_vals = [x[3] for x in valid_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.8
    x = range(len(iters))
    
    ax.bar(x, w_vals, bar_width, label="Wins", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x, d_vals, bar_width, bottom=w_vals, label="Draws", color="#95a5a6", edgecolor="black", linewidth=0.5)
    ax.bar(x, l_vals, bar_width, bottom=[w + d for w, d in zip(w_vals, d_vals)], label="Losses", color="#e74c3c", edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in iters], rotation=45, ha="right")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Games")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wld_comparison(
    graphs_dir: Path,
    iterations: List[int],
    wins: List[Optional[int]],
    losses: List[Optional[int]],
    draws: List[Optional[int]],
    title: str,
    filename: str,
) -> None:
    valid_data = [
        (i, w, l, d) for i, w, l, d in zip(iterations, wins, losses, draws)
        if w is not None and l is not None and d is not None
    ]
    if not valid_data:
        return
    
    iters = [x[0] for x in valid_data]
    w_vals = [x[1] for x in valid_data]
    l_vals = [x[2] for x in valid_data]
    d_vals = [x[3] for x in valid_data]
    
    net_wins = [w - l for w, l in zip(w_vals, l_vals)]
    colors = ["#2ecc71" if n > 0 else "#e74c3c" if n < 0 else "#95a5a6" for n in net_wins]
    
    height = max(4, min(len(iters) * 0.4, 12))
    fig, ax = plt.subplots(figsize=(10, height))
    
    bars = ax.barh(range(len(iters)), net_wins, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axvline(x=0, color="black", linewidth=1.5)
    ax.set_yticks(range(len(iters)))
    ax.set_yticklabels([str(i) for i in iters], fontsize=8)
    ax.set_ylabel("Iteration")
    ax.set_xlabel("Net Wins (Wins - Losses)")
    ax.set_title(title)
    
    max_abs = max(abs(n) for n in net_wins) if any(n != 0 for n in net_wins) else 10
    ax.set_xlim(-max_abs - max_abs * 0.3, max_abs + max_abs * 0.3)
    
    for bar, w, l, d in zip(bars, w_vals, l_vals, d_vals):
        x_pos = bar.get_width()
        if x_pos >= 0:
            ax.text(x_pos + max_abs * 0.05, bar.get_y() + bar.get_height()/2, 
                    f"{w}W/{l}L/{d}D", va="center", ha="left", fontsize=7)
        else:
            ax.text(x_pos - max_abs * 0.05, bar.get_y() + bar.get_height()/2, 
                    f"{w}W/{l}L/{d}D", va="center", ha="right", fontsize=7)
    
    ax.grid(True, alpha=0.3, axis="x")
    plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values, "r-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "total_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values, "m-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "policy_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vf_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, values, "c-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Function Loss")
    ax.set_title("Value Function Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "vf_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, vals, "y-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kl_divergence(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, vals, "orange", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "kl_divergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vf_explained_var(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, vals, "purple", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Value Function Explained Variance")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "vf_explained_var.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vf_loss_unclipped(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, vals, "teal", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("VF Loss (Unclipped)")
    ax.set_title("Value Function Loss (Unclipped)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "vf_loss_unclipped.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return
    
    for metric_name, plot_fn in PLOT_FUNCTIONS.items():
        if metric_name in metrics:
            values = metrics[metric_name]
            if any(v is not None for v in values):
                plot_fn(graphs_dir, iterations, values)
    
    if all(k in metrics for k in ["wins_vs_random", "losses_vs_random", "draws_vs_random"]):
        _plot_wld_stacked(
            graphs_dir, iterations,
            metrics["wins_vs_random"], metrics["losses_vs_random"], metrics["draws_vs_random"],
            "Results vs Random Agent", "results_vs_random.png"
        )
    
    if all(k in metrics for k in ["wins_vs_previous", "losses_vs_previous", "draws_vs_previous"]):
        _plot_wld_comparison(
            graphs_dir, iterations,
            metrics["wins_vs_previous"], metrics["losses_vs_previous"], metrics["draws_vs_previous"],
            "Performance vs Previous Checkpoint\n(Green = Net Positive, Red = Net Negative)",
            "results_vs_previous.png"
        )


def _filter_none(iterations: List[int], values: List[Optional[float]]) -> Tuple[List[int], List[float]]:
    filtered = [(i, v) for i, v in zip(iterations, values) if v is not None]
    if not filtered:
        return [], []
    return [x[0] for x in filtered], [x[1] for x in filtered]


PLOT_FUNCTIONS = {
    "episode_len_mean": plot_episode_length,
    "total_loss": plot_total_loss,
    "policy_loss": plot_policy_loss,
    "vf_loss": plot_vf_loss,
    "entropy": plot_entropy,
    "kl_divergence": plot_kl_divergence,
    "vf_explained_var": plot_vf_explained_var,
    "vf_loss_unclipped": plot_vf_loss_unclipped,
}