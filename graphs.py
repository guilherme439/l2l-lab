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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bar_width = 0.8
    x = range(len(iters))
    
    ax.bar(x, w_vals, bar_width, label="Wins", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x, d_vals, bar_width, bottom=w_vals, label="Draws", color="#95a5a6", edgecolor="black", linewidth=0.5)
    ax.bar(x, l_vals, bar_width, bottom=[w + d for w, d in zip(w_vals, d_vals)], label="Losses", color="#e74c3c", edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in iters], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Games", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wld_stacked_split(
    graphs_dir: Path,
    iterations: List[int],
    wins: List[Optional[int]],
    losses: List[Optional[int]],
    draws: List[Optional[int]],
    title_base: str,
    filename_base: str,
    split_interval: int,
) -> None:
    valid_data = [
        (i, w, l, d) for i, w, l, d in zip(iterations, wins, losses, draws)
        if w is not None and l is not None and d is not None
    ]
    if not valid_data:
        return
    
    max_iter = max(x[0] for x in valid_data)
    num_splits = (max_iter // split_interval) + 1
    
    for split_idx in range(num_splits):
        range_start = split_idx * split_interval
        range_end = (split_idx + 1) * split_interval
        
        split_data = [x for x in valid_data if range_start < x[0] <= range_end]
        if not split_data:
            continue
        
        iters = [x[0] for x in split_data]
        w_vals = [x[1] for x in split_data]
        l_vals = [x[2] for x in split_data]
        d_vals = [x[3] for x in split_data]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bar_width = 0.8
        x = range(len(iters))
        
        ax.bar(x, w_vals, bar_width, label="Wins", color="#2ecc71", edgecolor="black", linewidth=0.5)
        ax.bar(x, d_vals, bar_width, bottom=w_vals, label="Draws", color="#95a5a6", edgecolor="black", linewidth=0.5)
        ax.bar(x, l_vals, bar_width, bottom=[w + d for w, d in zip(w_vals, d_vals)], label="Losses", color="#e74c3c", edgecolor="black", linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}" for i in iters], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Games", fontsize=8)
        ax.set_title(f"{title_base} ({range_start}-{range_end})", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
        
        plt.tight_layout(pad=0.5)
        
        name, ext = filename_base.rsplit(".", 1)
        filename = f"{name}_{range_start}_{range_end}.{ext}"
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
    
    height = max(4, min(len(iters) * 0.35, 10))
    fig, ax = plt.subplots(figsize=(9, height))
    
    bars = ax.barh(range(len(iters)), net_wins, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axvline(x=0, color="black", linewidth=1.5)
    ax.set_yticks(range(len(iters)))
    ax.set_yticklabels([str(i) for i in iters], fontsize=6)
    ax.set_ylabel("Iteration", fontsize=8)
    ax.set_xlabel("Net Wins (Wins - Losses)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=6, pad=2)
    
    max_abs = max(abs(n) for n in net_wins) if any(n != 0 for n in net_wins) else 10
    ax.set_xlim(-max_abs - max_abs * 0.25, max_abs + max_abs * 0.25)
    
    for bar, w, l, d in zip(bars, w_vals, l_vals, d_vals):
        x_pos = bar.get_width()
        if x_pos >= 0:
            ax.text(x_pos + max_abs * 0.03, bar.get_y() + bar.get_height()/2, 
                    f"{w}W/{l}L/{d}D", va="center", ha="left", fontsize=5)
        else:
            ax.text(x_pos - max_abs * 0.03, bar.get_y() + bar.get_height()/2, 
                    f"{w}W/{l}L/{d}D", va="center", ha="right", fontsize=5)
    
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(pad=0.5)
    plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wld_comparison_split(
    graphs_dir: Path,
    iterations: List[int],
    wins: List[Optional[int]],
    losses: List[Optional[int]],
    draws: List[Optional[int]],
    title_base: str,
    filename_base: str,
    split_interval: int,
) -> None:
    valid_data = [
        (i, w, l, d) for i, w, l, d in zip(iterations, wins, losses, draws)
        if w is not None and l is not None and d is not None
    ]
    if not valid_data:
        return
    
    max_iter = max(x[0] for x in valid_data)
    num_splits = (max_iter // split_interval) + 1
    
    for split_idx in range(num_splits):
        range_start = split_idx * split_interval
        range_end = (split_idx + 1) * split_interval
        
        split_data = [x for x in valid_data if range_start < x[0] <= range_end]
        if not split_data:
            continue
        
        iters = [x[0] for x in split_data]
        w_vals = [x[1] for x in split_data]
        l_vals = [x[2] for x in split_data]
        d_vals = [x[3] for x in split_data]
        
        net_wins = [w - l for w, l in zip(w_vals, l_vals)]
        colors = ["#2ecc71" if n > 0 else "#e74c3c" if n < 0 else "#95a5a6" for n in net_wins]
        
        height = max(4, min(len(iters) * 0.35, 10))
        fig, ax = plt.subplots(figsize=(9, height))
        
        bars = ax.barh(range(len(iters)), net_wins, color=colors, edgecolor="black", linewidth=0.5)
        
        ax.axvline(x=0, color="black", linewidth=1.5)
        ax.set_yticks(range(len(iters)))
        ax.set_yticklabels([str(i) for i in iters], fontsize=6)
        ax.set_ylabel("Iteration", fontsize=8)
        ax.set_xlabel("Net Wins (Wins - Losses)", fontsize=8)
        ax.set_title(f"{title_base} ({range_start}-{range_end})", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=6, pad=2)
        
        max_abs = max(abs(n) for n in net_wins) if any(n != 0 for n in net_wins) else 10
        ax.set_xlim(-max_abs - max_abs * 0.25, max_abs + max_abs * 0.25)
        
        for bar, w, l, d in zip(bars, w_vals, l_vals, d_vals):
            x_pos = bar.get_width()
            if x_pos >= 0:
                ax.text(x_pos + max_abs * 0.03, bar.get_y() + bar.get_height()/2, 
                        f"{w}W/{l}L/{d}D", va="center", ha="left", fontsize=5)
            else:
                ax.text(x_pos - max_abs * 0.03, bar.get_y() + bar.get_height()/2, 
                        f"{w}W/{l}L/{d}D", va="center", ha="right", fontsize=5)
        
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(pad=0.5)
        
        name, ext = filename_base.rsplit(".", 1)
        filename = f"{name}_{range_start}_{range_end}.{ext}"
        plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_scatter(
    graphs_dir: Path,
    iterations: List[int],
    values: List[Optional[float]],
    ylabel: str,
    title: str,
    filename: str,
    color: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(iters, vals, c=color, s=8, alpha=0.6)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(graphs_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "Total Loss", "Total Loss", "total_loss.png", "red")


def plot_policy_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "Policy Loss", "Policy Loss", "policy_loss.png", "magenta")


def plot_vf_loss(graphs_dir: Path, iterations: List[int], values: List[float]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "Value Function Loss", "Value Function Loss", "vf_loss.png", "cyan")


def plot_entropy(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "Entropy", "Policy Entropy", "entropy.png", "gold")


def plot_kl_divergence(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    iters, vals = _filter_none(iterations, values)
    if not iters:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, vals, "orange", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("KL Divergence", fontsize=8)
    ax.set_title("KL Divergence", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(graphs_dir / "kl_divergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vf_explained_var(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "Explained Variance", "Value Function Explained Variance", 
                  "vf_explained_var.png", "purple", ylim=(-1, 1))


def plot_vf_loss_unclipped(graphs_dir: Path, iterations: List[int], values: List[Optional[float]]) -> None:
    _plot_scatter(graphs_dir, iterations, values, "VF Loss (Unclipped)", "Value Function Loss (Unclipped)", 
                  "vf_loss_unclipped.png", "teal")


def plot_metrics(graphs_dir: Path, metrics: Dict[str, List], eval_graph_split: int = 500) -> None:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return
    
    for metric_name, plot_fn in PLOT_FUNCTIONS.items():
        if metric_name in metrics:
            values = metrics[metric_name]
            if any(v is not None for v in values):
                plot_fn(graphs_dir, iterations, values)
    
    if all(k in metrics for k in ["wins_vs_random", "losses_vs_random", "draws_vs_random"]):
        _plot_wld_stacked_split(
            graphs_dir, iterations,
            metrics["wins_vs_random"], metrics["losses_vs_random"], metrics["draws_vs_random"],
            "Results vs Random Agent", "results_vs_random.png",
            eval_graph_split,
        )
    
    if all(k in metrics for k in ["wins_vs_previous", "losses_vs_previous", "draws_vs_previous"]):
        _plot_wld_comparison_split(
            graphs_dir, iterations,
            metrics["wins_vs_previous"], metrics["losses_vs_previous"], metrics["draws_vs_previous"],
            "Performance vs Previous Checkpoint",
            "results_vs_previous.png",
            eval_graph_split,
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
