import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .types import GameReport

if TYPE_CHECKING:
    from l2l_lab.training.metrics_store import EvalSeries

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


@dataclass
class StampedReport:
    """A ``GameReport`` paired with the eval context the caller knew."""
    iteration: int
    eval_label: str
    as_position: str
    report: GameReport


def render_report(
    run_name: str,
    iteration: int,
    backend_name: str,
    env_name: str,
    scalars: dict[str, Any],
    evaluations: dict[str, dict[str, EvalSeries]],
    probe_results: list[dict[str, Any]],
    reports: list[StampedReport],
    sparkline_window: int,
) -> str:
    """Render a full snapshot report to a Markdown string.

    Empty sections are omitted entirely rather than printing placeholder text,
    so reports stay tight for envs/backends that don't populate every source.
    """
    parts: list[str] = []
    parts.append(_render_header(run_name, iteration, backend_name, env_name))

    scalar_section = _render_scalar_metrics(scalars, sparkline_window)
    if scalar_section:
        parts.append(scalar_section)

    eval_section = _render_evaluations(evaluations)
    if eval_section:
        parts.append(eval_section)

    probe_section = _render_probe_states(probe_results)
    if probe_section:
        parts.append(probe_section)

    sample_section = _render_sample_games(reports)
    if sample_section:
        parts.append(sample_section)

    return "\n\n".join(parts) + "\n"


def sparkline(values: list[Optional[float]]) -> str:
    """Render a numeric series as a compact Unicode sparkline. Missing (None)
    values render as spaces so gaps are visible without breaking alignment.
    """
    numeric = [v for v in values if v is not None and _is_finite(v)]
    if not numeric:
        return ""
    lo, hi = min(numeric), max(numeric)
    span = hi - lo
    out: list[str] = []
    for v in values:
        if v is None or not _is_finite(v):
            out.append(" ")
            continue
        if span == 0:
            out.append(_SPARK_CHARS[len(_SPARK_CHARS) // 2])
        else:
            idx = int(round((v - lo) / span * (len(_SPARK_CHARS) - 1)))
            idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
            out.append(_SPARK_CHARS[idx])
    return "".join(out)


def _render_header(run_name: str, iteration: int, backend_name: str, env_name: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"# Report: {run_name} @ iter {iteration}\n"
        f"backend={backend_name} env={env_name} generated={now}"
    )


def _render_scalar_metrics(metrics: dict[str, Any], window: int) -> str:
    lines: list[str] = []
    for key in sorted(metrics.keys()):
        if key in ("iteration", "evaluations"):
            continue
        values = metrics[key]
        if not isinstance(values, list):
            continue
        numeric: list[Optional[float]] = []
        for v in values:
            if v is None:
                numeric.append(None)
                continue
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                numeric = []
                break
        if not numeric:
            continue
        recent = numeric[-window:]
        finite_recent = [v for v in recent if v is not None and _is_finite(v)]
        if len(finite_recent) < 2:
            continue
        latest = next((v for v in reversed(numeric) if v is not None and _is_finite(v)), None)
        if latest is None:
            continue
        lines.append(f"- `{key}` latest={_fmt(latest)} spark={sparkline(recent)}")

    if not lines:
        return ""
    return "## Scalar metrics\n" + "\n".join(lines)


def _render_evaluations(evaluations: dict[str, dict[str, EvalSeries]]) -> str:
    headings = {"training": "Training Evals", "checkpoint": "Checkpoint Evals"}
    sections: list[str] = []
    for eval_type, heading in headings.items():
        series_by_label = evaluations.get(eval_type)
        if not series_by_label:
            continue

        lines: list[str] = []
        for label in sorted(series_by_label.keys()):
            series = series_by_label[label]
            eval_lines: list[str] = []
            for position, points in (("as_p0", series.as_p0), ("as_p1", series.as_p1)):
                if not points:
                    continue
                latest = points[-1]
                total = latest.wins + latest.losses + latest.draws
                win_rate = latest.wins / total if total > 0 else 0.0
                eval_lines.append(
                    f"  - {position}: {latest.wins}W/{latest.losses}L/{latest.draws}D"
                    f" (win_rate={win_rate:.1%})"
                )

            if eval_lines:
                lines.append(f"- **{label}**")
                lines.extend(eval_lines)

        if lines:
            sections.append(f"### {heading}\n" + "\n".join(lines))

    if not sections:
        return ""
    return "## Evaluations\n" + "\n\n".join(sections)


def _render_probe_states(probe_results: list[dict[str, Any]]) -> str:
    if not probe_results:
        return ""

    lines: list[str] = []
    for probe in probe_results:
        lines.append(f"### {probe['label']}")
        if probe.get("description"):
            lines.append(f"_{probe['description']}_")
        if probe.get("current_player"):
            lines.append(f"current_player: `{probe['current_player']}`")
        lines.append(f"legal_actions: {probe.get('legal_actions', [])}")

        policy = probe.get("policy") or []
        legal = set(probe.get("legal_actions") or [])
        policy_parts = []
        for i, p in enumerate(policy):
            marker = "*" if i in legal else " "
            policy_parts.append(f"{marker}a{i}={p:.3f}")
        lines.append("policy: " + "  ".join(policy_parts))

        value = probe.get("value")
        lines.append(f"value: {_fmt(value)}")
        lines.append(
            f"logits: min={_fmt(probe.get('logits_min'))} "
            f"max={_fmt(probe.get('logits_max'))} "
            f"mean={_fmt(probe.get('logits_mean'))} "
            f"std={_fmt(probe.get('logits_std'))}"
        )
        lines.append("")

    return "## Probe states\n" + "\n".join(lines).rstrip()


def _render_sample_games(reports: list[StampedReport]) -> str:
    if not reports:
        return ""

    lines: list[str] = []
    for stamped in reports:
        r = stamped.report
        result_str = {1: "p0 win", -1: "p1 win", 0: "draw"}.get(r.result_from_p0, "?")
        header = (
            f"### {stamped.eval_label} | {stamped.as_position} | "
            f"iter={stamped.iteration} | p0={r.p0_name} vs p1={r.p1_name} | "
            f"result={result_str} | moves={r.num_moves}"
        )
        lines.append(header)
        move_strs = []
        for idx, (agent_id, action, _obs) in enumerate(r.moves, start=1):
            move_strs.append(f"{idx}.{agent_id}:{action}")
        if move_strs:
            lines.append(" ".join(move_strs))
        lines.append("")

    return "## Sample games\n" + "\n".join(lines).rstrip()


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not _is_finite(xf):
        return str(xf)
    if abs(xf) >= 1000 or (xf != 0 and abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:.4f}"


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
