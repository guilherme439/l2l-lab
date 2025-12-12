from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _filter_none(iterations: List[int], values: List[Optional[float]]) -> Tuple[List[int], List[float]]:
    filtered = [(i, v) for i, v in zip(iterations, values) if v is not None]
    if not filtered:
        return [], []
    return [x[0] for x in filtered], [x[1] for x in filtered]


def _has_valid_data(values: List[Optional[float]]) -> bool:
    valid = [v for v in values if v is not None]
    return len(valid) >= 2


def _has_variation(values: List[Optional[float]], threshold: float = 1e-9) -> bool:
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return False
    return (max(valid) - min(valid)) > threshold


def _rolling_mean(values: List[float], window: int = 10) -> List[float]:
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot_training_overview(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    ep_len = metrics.get("episode_len_mean", [])
    ep_reward = metrics.get("episode_reward_mean", [])
    
    if not _has_valid_data(ep_len):
        return
    
    iters_len, vals_len = _filter_none(iterations, ep_len)
    has_reward = _has_valid_data(ep_reward) and _has_variation(ep_reward)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    smoothed_len = _rolling_mean(vals_len, window=15)
    ax1.plot(iters_len, vals_len, color="#3498db", alpha=0.3, linewidth=0.8, label="Episode Length")
    ax1.plot(iters_len, smoothed_len, color="#2980b9", linewidth=2, label="Episode Length (smoothed)")
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Episode Length", color="#2980b9", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.grid(True, alpha=0.3)
    
    if has_reward:
        iters_rew, vals_rew = _filter_none(iterations, ep_reward)
        smoothed_rew = _rolling_mean(vals_rew, window=15)
        ax2 = ax1.twinx()
        ax2.plot(iters_rew, vals_rew, color="#e74c3c", alpha=0.3, linewidth=0.8)
        ax2.plot(iters_rew, smoothed_rew, color="#c0392b", linewidth=2, label="Episode Reward (smoothed)")
        ax2.set_ylabel("Episode Reward", color="#c0392b", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#c0392b")
    
    ax1.set_title("Training Progress", fontsize=12, fontweight="bold")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_reward:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax1.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "training_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_breakdown(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    total_loss = metrics.get("total_loss", [])
    policy_loss = metrics.get("policy_loss", [])
    vf_loss = metrics.get("vf_loss", [])
    
    iters_t, vals_t = _filter_none(iterations, total_loss)
    iters_p, vals_p = _filter_none(iterations, policy_loss)
    iters_v, vals_v = _filter_none(iterations, vf_loss)
    
    if vals_t:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(iters_t, vals_t, color="#e74c3c", s=8, alpha=0.6)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Total Loss", fontsize=10)
        ax.set_title("Total Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "total_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    if vals_p:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(iters_p, vals_p, color="#9b59b6", s=8, alpha=0.6)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Policy Loss", fontsize=10)
        ax.set_title("Policy Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "policy_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    if vals_v:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(iters_v, vals_v, color="#3498db", s=8, alpha=0.6)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Value Loss", fontsize=10)
        ax.set_title("Value Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_dir / "value_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_policy_health(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    entropy = metrics.get("entropy", [])
    kl_div = metrics.get("kl_divergence", [])
    
    has_entropy = _has_valid_data(entropy)
    has_kl = _has_valid_data(kl_div) and _has_variation(kl_div)
    
    if not has_entropy and not has_kl:
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    if has_entropy:
        iters_e, vals_e = _filter_none(iterations, entropy)
        ax1.scatter(iters_e, vals_e, color="#f39c12", s=8, alpha=0.6, label="Entropy")
        ax1.set_ylabel("Entropy", color="#f39c12", fontsize=10)
        ax1.tick_params(axis="y", labelcolor="#f39c12")
    
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    if has_kl:
        iters_k, vals_k = _filter_none(iterations, kl_div)
        ax2 = ax1.twinx() if has_entropy else ax1
        ax2.plot(
            iters_k,
            vals_k,
            color="#16a085",
            linewidth=1,
            alpha=0.8,
            linestyle="-",
            marker="o",
            markersize=1,
            label="KL Divergence",
        )
        ax2.set_ylabel("KL Divergence", color="#16a085", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#16a085")
    
    ax1.set_title("Policy Health" + (" (Entropy + KL)" if has_kl else " (Entropy)"), fontsize=12, fontweight="bold")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_kl and has_entropy:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    else:
        ax1.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "policy_health.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_value_function(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    vf_loss = metrics.get("vf_loss", [])
    vf_explained = metrics.get("vf_explained_var", [])
    
    has_loss = _has_valid_data(vf_loss)
    has_explained = _has_valid_data(vf_explained)
    
    if not has_loss and not has_explained:
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    if has_loss:
        iters_l, vals_l = _filter_none(iterations, vf_loss)
        ax1.plot(iters_l, vals_l, color="#3498db", linewidth=1.5, label="VF Loss")
        ax1.set_ylabel("VF Loss", color="#3498db", fontsize=10)
        ax1.tick_params(axis="y", labelcolor="#3498db")
    
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    if has_explained:
        iters_e, vals_e = _filter_none(iterations, vf_explained)
        ax2 = ax1.twinx() if has_loss else ax1
        ax2.plot(iters_e, vals_e, color="#27ae60", linewidth=1.5, linestyle="--", label="Explained Variance")
        ax2.set_ylabel("Explained Variance", color="#27ae60", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#27ae60")
        ax2.set_ylim(-1.1, 1.1)
    
    ax1.set_title("Value Function Quality", fontsize=12, fontweight="bold")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_explained and has_loss:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    else:
        ax1.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "value_function.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    lr = metrics.get("learning_rate", [])
    
    if not _has_valid_data(lr):
        return
    
    if not _has_variation(lr):
        return
    
    iters, vals = _filter_none(iterations, lr)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, vals, color="#1abc9c", linewidth=2)
    ax.fill_between(iters, 0, vals, color="#1abc9c", alpha=0.2)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_icm_dashboard(graphs_dir: Path, metrics: Dict[str, List]) -> None:
    iterations = metrics.get("iteration", [])
    intrinsic = metrics.get("intrinsic_reward_mean", [])
    forward_loss = metrics.get("icm_forward_loss", [])
    inverse_loss = metrics.get("icm_inverse_loss", [])
    
    has_intrinsic = _has_valid_data(intrinsic)
    has_forward = _has_valid_data(forward_loss)
    has_inverse = _has_valid_data(inverse_loss)
    
    if not has_intrinsic and not has_forward:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    if has_intrinsic:
        iters_i, vals_i = _filter_none(iterations, intrinsic)
        axes[0].plot(iters_i, vals_i, color="#e74c3c", linewidth=1.5, label="Intrinsic Reward")
        axes[0].set_ylabel("Intrinsic Reward", fontsize=10)
        axes[0].set_title("Curiosity Module - Intrinsic Rewards", fontsize=11, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No intrinsic reward data", ha="center", va="center", transform=axes[0].transAxes)
    
    if has_forward or has_inverse:
        if has_forward:
            iters_f, vals_f = _filter_none(iterations, forward_loss)
            axes[1].plot(iters_f, vals_f, color="#3498db", linewidth=1.5, label="Forward Loss")
        if has_inverse:
            iters_inv, vals_inv = _filter_none(iterations, inverse_loss)
            axes[1].plot(iters_inv, vals_inv, color="#9b59b6", linewidth=1.5, label="Inverse Loss")
        axes[1].set_ylabel("Loss", fontsize=10)
        axes[1].set_title("ICM Losses (Forward & Inverse)", fontsize=11, fontweight="bold")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No ICM loss data", ha="center", va="center", transform=axes[1].transAxes)
    
    axes[1].set_xlabel("Iteration", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(graphs_dir / "icm_dashboard.png", dpi=150, bbox_inches="tight")
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


def plot_vs_previous_performance(graphs_dir: Path, metrics: Dict[str, List], split_interval: int = 500) -> None:
    iterations = metrics.get("iteration", [])
    wins = metrics.get("wins_vs_previous", [])
    losses = metrics.get("losses_vs_previous", [])
    draws = metrics.get("draws_vs_previous", [])
    
    valid_data = [
        (i, w, l, d) for i, w, l, d in zip(iterations, wins, losses, draws)
        if w is not None and l is not None and d is not None
    ]
    
    if len(valid_data) < 2:
        return
    
    iters = [x[0] for x in valid_data]
    w_vals = [x[1] for x in valid_data]
    l_vals = [x[2] for x in valid_data]
    d_vals = [x[3] for x in valid_data]
    
    net_wins = [w - l for w, l in zip(w_vals, l_vals)]
    
    max_iter = max(iters)
    num_splits = (max_iter // split_interval) + 1
    
    for split_idx in range(num_splits):
        range_start = split_idx * split_interval
        range_end = (split_idx + 1) * split_interval
        
        split_indices = [i for i, it in enumerate(iters) if range_start < it <= range_end]
        if not split_indices:
            continue
        
        split_iters = [iters[i] for i in split_indices]
        split_net = [net_wins[i] for i in split_indices]
        split_w = [w_vals[i] for i in split_indices]
        split_l = [l_vals[i] for i in split_indices]
        split_d = [d_vals[i] for i in split_indices]
        
        colors = ["#2ecc71" if n > 0 else "#e74c3c" if n < 0 else "#95a5a6" for n in split_net]
        
        height = max(4, min(len(split_iters) * 0.4, 12))
        fig, ax = plt.subplots(figsize=(10, height))
        
        bars = ax.barh(range(len(split_iters)), split_net, color=colors, edgecolor="black", linewidth=0.5)
        
        ax.axvline(x=0, color="black", linewidth=1.5)
        ax.set_yticks(range(len(split_iters)))
        ax.set_yticklabels([str(i) for i in split_iters], fontsize=8)
        ax.set_ylabel("Iteration", fontsize=10)
        ax.set_xlabel("Net Wins (Wins - Losses)", fontsize=10)
        ax.set_title(f"Performance vs Previous Checkpoint ({range_start}-{range_end})", fontsize=12, fontweight="bold")
        
        max_abs = max(abs(n) for n in split_net) if any(n != 0 for n in split_net) else 10
        ax.set_xlim(-max_abs - max_abs * 0.3, max_abs + max_abs * 0.3)
        
        for bar, w, l, d in zip(bars, split_w, split_l, split_d):
            x_pos = bar.get_width()
            text = f"{w}W / {l}L / {d}D"
            if x_pos >= 0:
                ax.text(x_pos + max_abs * 0.05, bar.get_y() + bar.get_height()/2, text, va="center", ha="left", fontsize=7)
            else:
                ax.text(x_pos - max_abs * 0.05, bar.get_y() + bar.get_height()/2, text, va="center", ha="right", fontsize=7)
        
        ax.grid(True, alpha=0.3, axis="x")
        
        plt.tight_layout()
        plt.savefig(graphs_dir / f"vs_previous_{range_start}_{range_end}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_metrics(graphs_dir: Path, metrics: Dict[str, List], eval_graph_split: int = 500) -> None:
    iterations = metrics.get("iteration", [])
    if not iterations:
        return
    
    plot_training_overview(graphs_dir, metrics)
    plot_loss_breakdown(graphs_dir, metrics)
    plot_policy_health(graphs_dir, metrics)
    plot_value_function(graphs_dir, metrics)
    plot_learning_rate(graphs_dir, metrics)
    
    has_icm = _has_valid_data(metrics.get("intrinsic_reward_mean", []))
    if has_icm:
        plot_icm_dashboard(graphs_dir, metrics)
    
    has_vs_random = all(
        k in metrics and _has_valid_data(metrics[k])
        for k in ["wins_vs_random", "losses_vs_random", "draws_vs_random"]
    )
    if has_vs_random:
        _plot_wld_stacked_split(
            graphs_dir, iterations,
            metrics["wins_vs_random"], metrics["losses_vs_random"], metrics["draws_vs_random"],
            "Results vs Random Agent", "results_vs_random.png",
            eval_graph_split,
        )
    
    has_vs_prev = all(
        k in metrics and _has_valid_data(metrics[k])
        for k in ["wins_vs_previous", "losses_vs_previous", "draws_vs_previous"]
    )
    if has_vs_prev:
        plot_vs_previous_performance(graphs_dir, metrics, eval_graph_split)
