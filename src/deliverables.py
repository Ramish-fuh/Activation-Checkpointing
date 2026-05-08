#!/usr/bin/env python3
"""Generate the final deliverable plots and summary tables.

Outputs:
- Peak memory consumption vs mini-batch size, with and without AC.
- Forward peak memory consumption vs mini-batch size, with and without AC.
- Iteration latency vs mini-batch size, with and without AC.
- A compact table-style summary of profiling and static analysis stats.
- A JSON summary with profiling and checkpoint diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.utils._pytree as pytree

sys.path.insert(0, str(Path(__file__).parent))

from activation_checkpoint import apply_checkpoint_plan, smoke_check_graph
from benchmarks import (
    Experiment,
    _build_checkpoint_diagnostics,
    _build_phase1_diagnostics,
    _fmt_mb,
    _plots_dir,
)
from graph_prof import GraphProfiler
from graph_tracer import _compile
from mu_two_policy import PolicyConfig, build_checkpoint_plan, validate_checkpoint_plan


def _ensure_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to generate the deliverable plots") from exc

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
        import importlib

        importlib.reload(plt)
    return plt


def _parse_budget_fraction(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    return float(value)


def _profile_experiment(model_name: str, batch_size: int, budget_fraction: float) -> Dict[str, Any]:
    experiment = Experiment(model_name, batch_size)
    experiment.init_opt_states()

    def _log(message: str) -> None:
        print(f"[{model_name} bs={batch_size}] {message}", flush=True)

    _log("starting graph compilation")
    compiled = _compile(
        experiment.train_step,
        experiment.model,
        experiment.optimizer,
        experiment.example_inputs,
    )
    _log("graph compiled; beginning profiler warmup")
    baseline_gm = copy.deepcopy(compiled.gm)
    flat_inputs = compiled.flat_state + pytree.tree_flatten(
        [(experiment.model, experiment.optimizer, experiment.example_inputs), {}]
    )[0]

    profiler = GraphProfiler(compiled.gm)
    with torch.no_grad():
        for _ in range(2):
            profiler.run(*flat_inputs)
        _log("warmup complete; collecting profiling runs")
        profiler.reset_stats()
        for _ in range(3):
            profiler.run(*flat_inputs)
        profiler.aggregate_stats()
    _log("profiling complete; building checkpoint plan")

    diagnostics = _build_phase1_diagnostics(profiler)

    config = PolicyConfig(default_memory_budget_fraction=budget_fraction)
    plan = build_checkpoint_plan(profiler, config)
    _log(f"checkpoint plan built with {len(plan.recompute_nodes)} recompute nodes; validating")
    report = validate_checkpoint_plan(profiler, plan, config)
    diagnostics["checkpoint"] = _build_checkpoint_diagnostics(profiler, plan, report)

    if not report.ok:
        raise RuntimeError(report.format_summary())
    _log("checkpoint plan validated")

    checkpoint_gm = copy.deepcopy(compiled.gm)
    if plan.recompute_nodes:
        _log("applying checkpoint rewrite and running smoke check")
        checkpoint_gm = apply_checkpoint_plan(checkpoint_gm, plan)
        smoke_ok, smoke_message = smoke_check_graph(
            checkpoint_gm,
            flat_inputs,
            reference_gm=compiled.gm,
        )
        diagnostics["checkpoint"]["smoke_check"] = {
            "ok": smoke_ok,
            "message": smoke_message,
        }
        if not smoke_ok:
            raise RuntimeError(f"Checkpoint smoke check failed: {smoke_message}")
        _log("smoke check passed")

    _log("measuring latency comparison")
    latency = experiment._measure_latency_comparison(baseline_gm, checkpoint_gm, flat_inputs)
    _log("latency measurement complete")

    return {
        "model_name": model_name,
        "batch_size": batch_size,
        "budget_fraction": float(budget_fraction),
        "diagnostics": diagnostics,
        "checkpoint_report": report.to_dict(),
        "latency": latency,
        "peak_memory_without_ac_bytes": int(report.overall_peak_before_bytes),
        "peak_memory_with_ac_bytes": int(report.overall_peak_after_bytes),
        "peak_memory_without_ac_mb": _fmt_mb(int(report.overall_peak_before_bytes)),
        "peak_memory_with_ac_mb": _fmt_mb(int(report.overall_peak_after_bytes)),
        "forward_peak_without_ac_bytes": int(report.forward_peak_before_bytes),
        "forward_peak_with_ac_bytes": int(report.forward_peak_after_bytes),
        "forward_peak_without_ac_mb": _fmt_mb(int(report.forward_peak_before_bytes)),
        "forward_peak_with_ac_mb": _fmt_mb(int(report.forward_peak_after_bytes)),
        "latency_without_ac_ms": float(latency["baseline_ms"]),
        "latency_with_ac_ms": float(latency["checkpoint_ms"]),
        "latency_overhead_ms": float(latency["overhead_ms"]),
        "latency_overhead_percent": latency["overhead_percent"],
    }


def _save_peak_memory_plot(results: Sequence[Dict[str, Any]], save_path: str, model_name: str) -> None:
    plt = _ensure_matplotlib()
    batch_sizes = [row["batch_size"] for row in results]
    baseline = [row["peak_memory_without_ac_mb"] for row in results]
    checkpoint = [row["peak_memory_with_ac_mb"] for row in results]

    x = list(range(len(batch_sizes)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_bars = ax.bar([i - width / 2 for i in x], baseline, width=width, label="Without AC", color="#1f77b4")
    checkpoint_bars = ax.bar([i + width / 2 for i in x], checkpoint, width=width, label="With AC", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Overall Peak Memory vs Mini-batch Size -- {model_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (baseline_bars, checkpoint_bars):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_forward_peak_plot(results: Sequence[Dict[str, Any]], save_path: str, model_name: str) -> None:
    plt = _ensure_matplotlib()
    batch_sizes = [row["batch_size"] for row in results]
    baseline = [row["forward_peak_without_ac_mb"] for row in results]
    checkpoint = [row["forward_peak_with_ac_mb"] for row in results]

    x = list(range(len(batch_sizes)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_bars = ax.bar([i - width / 2 for i in x], baseline, width=width, label="Without AC", color="#1f77b4")
    checkpoint_bars = ax.bar([i + width / 2 for i in x], checkpoint, width=width, label="With AC", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Forward peak memory (MB)")
    ax.set_title(f"Forward Peak Memory vs Mini-batch Size -- {model_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (baseline_bars, checkpoint_bars):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_latency_plot(results: Sequence[Dict[str, Any]], save_path: str, model_name: str) -> None:
    plt = _ensure_matplotlib()
    batch_sizes = [row["batch_size"] for row in results]
    baseline = [row["latency_without_ac_ms"] for row in results]
    checkpoint = [row["latency_with_ac_ms"] for row in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(batch_sizes, baseline, marker="o", linewidth=2.2, label="Without AC", color="#1f77b4")
    ax.plot(batch_sizes, checkpoint, marker="o", linewidth=2.2, label="With AC", color="#d62728")
    ax.set_xlabel("Mini-batch size")
    ax.set_ylabel("Iteration latency (ms)")
    ax.set_title(f"Iteration Latency vs Mini-batch Size -- {model_name}")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_summary_table(results: Sequence[Dict[str, Any]], save_path: str, model_name: str) -> None:
    plt = _ensure_matplotlib()

    headers = [
        "Batch",
        "Fwd peak\nwo AC",
        "Fwd peak\nAC",
        "Overall peak\nwo AC",
        "Overall peak\nAC",
        "Latency\nwo AC",
        "Latency\nAC",
        "Nodes",
        "ACT",
        "Recompute",
        "Budget",
    ]

    rows = []
    for row in results:
        diag = row["diagnostics"]
        classification = diag["classification"]
        checkpoint = diag["checkpoint"]
        rows.append([
            str(row["batch_size"]),
            f"{row['forward_peak_without_ac_mb']:.1f}",
            f"{row['forward_peak_with_ac_mb']:.1f}",
            f"{row['peak_memory_without_ac_mb']:.1f}",
            f"{row['peak_memory_with_ac_mb']:.1f}",
            f"{row['latency_without_ac_ms']:.1f}",
            f"{row['latency_with_ac_ms']:.1f}",
            str(classification["total_nodes"]),
            str(classification["node_type_counts"]["ACT"]),
            str(checkpoint["recompute_nodes"]),
            "yes" if checkpoint["memory_budget_satisfied"] else "no",
        ])

    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#fafafa")

    ax.set_title(f"Profiling and Static Analysis Summary -- {model_name}", pad=18)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_deliverables(
    model_name: str,
    batch_sizes: Sequence[int],
    budget_fraction: float = 0.5,
) -> Dict[str, Any]:
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")

    plots_dir = _plots_dir()
    os.makedirs(plots_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        print(f"Running deliverable sweep for {model_name} bs={batch_size}", flush=True)
        results.append(_profile_experiment(model_name, batch_size, budget_fraction))

    peak_plot_path = os.path.join(plots_dir, f"deliverable_peak_memory_vs_batch_size_{model_name}.png")
    forward_peak_plot_path = os.path.join(plots_dir, f"deliverable_forward_peak_vs_batch_size_{model_name}.png")
    latency_plot_path = os.path.join(plots_dir, f"deliverable_iteration_latency_vs_batch_size_{model_name}.png")
    summary_table_path = os.path.join(plots_dir, f"deliverable_stats_table_{model_name}.png")
    summary_path = os.path.join(plots_dir, f"deliverable_summary_{model_name}.json")

    _save_peak_memory_plot(results, peak_plot_path, model_name)
    _save_forward_peak_plot(results, forward_peak_plot_path, model_name)
    _save_latency_plot(results, latency_plot_path, model_name)
    _save_summary_table(results, summary_table_path, model_name)

    summary = {
        "model_name": model_name,
        "batch_sizes": list(batch_sizes),
        "budget_fraction": float(budget_fraction),
        "results": results,
        "artifacts": {
            "peak_memory_plot": os.path.abspath(peak_plot_path),
            "forward_peak_plot": os.path.abspath(forward_peak_plot_path),
            "latency_plot": os.path.abspath(latency_plot_path),
            "stats_table_plot": os.path.abspath(summary_table_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {os.path.abspath(summary_path)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate profiling and checkpoint deliverables")
    parser.add_argument("model_name", help="Model name to benchmark")
    parser.add_argument("batch_sizes", nargs="+", type=int, help="One or more batch sizes")
    parser.add_argument(
        "--budget",
        default="50%",
        help="Memory budget as a fraction (0.5) or percent string (50%%)",
    )
    args = parser.parse_args()

    budget_fraction = _parse_budget_fraction(args.budget)
    if budget_fraction < 0 or budget_fraction > 1:
        raise SystemExit("--budget must resolve to a value between 0 and 1")

    generate_deliverables(args.model_name, args.batch_sizes, budget_fraction=budget_fraction)
