# ============================================================================
# benchmarks.py
#
# Experiment harness for tracing and profiling a single training step.
# Supports BERT and Resnet152 architectures. The
# graph_transformation callback runs the GraphProfiler and saves a
# peak-memory breakdown plot after the first compiled iteration.
# ============================================================================

import os
import sys
import json
import copy
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torchvision.models import resnet152
from graph_prof import GraphProfiler, NodeType
from activation_checkpoint import clone_graph_inputs
from graph_tracer import SEPFunction, compile

try:
    from transformers import BertConfig, BertForMaskedLM
except ImportError:
    BertConfig = None
    BertForMaskedLM = None


def _plots_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")


def _fmt_mb(num_bytes: int) -> float:
    return num_bytes / 1024**2


def _node_type_counts(profiler: GraphProfiler) -> Dict[str, int]:
    counts: Dict[str, int] = {nt.name: 0 for nt in NodeType}
    for node_type in profiler.node_type.values():
        counts[node_type.name] += 1
    return counts


def _build_phase1_diagnostics(profiler: GraphProfiler) -> Dict[str, Any]:
    decomposed = profiler._find_decomposed_parents()
    alive = profiler._build_alive_ranges(decomposed)
    steps, by_type_series, totals = profiler._build_memory_timeline(alive)

    forward_peak_step, forward_peak_bytes, forward_breakdown = profiler._sweep_for_forward_peak(alive)
    overall_peak_bytes, overall_breakdown = profiler.compute_peak_memory()

    total_matches_components = all(
        total == sum(by_type.values())
        for total, by_type in zip(totals, by_type_series)
    )

    def _breakdown_to_bytes(breakdown: Dict[NodeType, int]) -> Dict[str, int]:
        return {nt.name: int(breakdown.get(nt, 0)) for nt in NodeType}

    def _top_other_alive(step_index: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for node, (born, dies) in alive.items():
            if born <= step_index <= dies and profiler.node_type.get(node, NodeType.OTHER) is NodeType.OTHER:
                mem_bytes = int(profiler.node_mem_bytes.get(node.name, 0))
                if mem_bytes <= 0:
                    continue
                rows.append(
                    {
                        "name": node.name,
                        "memory_bytes": mem_bytes,
                        "memory_mb": _fmt_mb(mem_bytes),
                        "region": profiler.node_region.get(node, "?"),
                    }
                )
        rows.sort(key=lambda row: row["memory_bytes"], reverse=True)
        return rows[:20]

    checkpointable_acts = [n for n in profiler.intermediate_nodes if profiler.first_bw_access.get(n) is not None]
    forward_peak_node = profiler.node_list[forward_peak_step].name if 0 <= forward_peak_step < len(profiler.node_list) else None
    forward_peak_region = profiler.node_region.get(profiler.node_list[forward_peak_step], "?") if 0 <= forward_peak_step < len(profiler.node_list) else "?"
    overall_peak_step = max(range(len(totals)), key=lambda idx: totals[idx]) if totals else -1
    overall_peak_node = profiler.node_list[overall_peak_step].name if 0 <= overall_peak_step < len(profiler.node_list) else None

    return {
        "accounting_invariants": {
            "timeline_total_equals_sum_of_components": total_matches_components,
            "forward_peak_total_equals_sum_of_components": forward_peak_bytes == sum(forward_breakdown.values()),
            "overall_peak_total_equals_sum_of_components": overall_peak_bytes == sum(overall_breakdown.values()),
        },
        "classification": {
            "total_nodes": len(profiler.node_list),
            "node_type_counts": _node_type_counts(profiler),
            "checkpointable_activations": len(checkpointable_acts),
            "forward_candidates": sum(1 for node in profiler.node_list if profiler._is_forward_candidate(node)),
        },
        "forward_peak": {
            "step_index": forward_peak_step,
            "op_name": forward_peak_node,
            "region": forward_peak_region,
            "total_bytes": forward_peak_bytes,
            "total_mb": _fmt_mb(forward_peak_bytes),
            "breakdown_bytes": _breakdown_to_bytes(forward_breakdown),
            "top_other_alive": _top_other_alive(forward_peak_step),
        },
        "overall_peak": {
            "step_index": overall_peak_step,
            "op_name": overall_peak_node,
            "total_bytes": overall_peak_bytes,
            "total_mb": _fmt_mb(overall_peak_bytes),
            "breakdown_bytes": _breakdown_to_bytes(overall_breakdown),
        },
        "timeline": {
            "num_steps": len(steps),
            "peak_total_bytes": max(totals, default=0),
            "peak_total_mb": _fmt_mb(max(totals, default=0)),
        },
    }


def _save_other_memory_plot(profiler: GraphProfiler, save_path: str, title: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed -- skipping OTHER memory plot")
        return

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
        import importlib
        importlib.reload(plt)

    decomposed = profiler._find_decomposed_parents()
    alive = profiler._build_alive_ranges(decomposed)
    forward_peak_step, _, _ = profiler._sweep_for_forward_peak(alive)

    rows: List[Dict[str, Any]] = []
    for node, (born, dies) in alive.items():
        if born <= forward_peak_step <= dies and profiler.node_type.get(node, NodeType.OTHER) is NodeType.OTHER:
            mem_bytes = int(profiler.node_mem_bytes.get(node.name, 0))
            if mem_bytes <= 0:
                continue
            rows.append(
                {
                    "name": node.name,
                    "memory_bytes": mem_bytes,
                    "memory_mb": _fmt_mb(mem_bytes),
                    "region": profiler.node_region.get(node, "?"),
                }
            )

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if not rows:
        ax.text(0.5, 0.5, "No OTHER tensors alive at forward peak", ha="center", va="center")
        ax.set_axis_off()
    else:
        rows.sort(key=lambda row: row["memory_bytes"], reverse=True)
        rows = rows[:20]
        labels = [row["name"] for row in reversed(rows)]
        values = [row["memory_mb"] for row in reversed(rows)]
        ax.barh(labels, values, color="#9467bd")
        ax.set_xlabel("Memory (MB)")
        ax.set_ylabel("Tensor")
        ax.grid(axis="x", alpha=0.25)

    ax.set_title(f"OTHER tensors alive at forward peak{f' -- {title}' if title else ''}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {os.path.abspath(save_path)}")
    plt.close(fig)


def _save_gradient_accumulation_plot(profiler: GraphProfiler, save_path: str, title: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed -- skipping gradient accumulation plot")
        return

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
        import importlib
        importlib.reload(plt)

    decomposed = profiler._find_decomposed_parents()
    alive = profiler._build_alive_ranges(decomposed)
    steps, by_type_series, _ = profiler._build_memory_timeline(alive)
    op_counts = [step + 1 for step in steps]
    grad_mb = [series.get(NodeType.GRAD, 0) / 1024**2 for series in by_type_series]
    param_mb = [series.get(NodeType.PARAM, 0) / 1024**2 for series in by_type_series]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.step(op_counts, grad_mb, where="post", linewidth=2.0, color="#ff7f0e", label="gradients")
    ax.plot(op_counts, param_mb, linewidth=1.5, color="#1f77b4", alpha=0.9, label="parameters")
    ax.set_xlabel("Operations")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"Gradient accumulation over operations{f' -- {title}' if title else ''}")
    ax.grid(alpha=0.3)
    ax.legend()
    if op_counts:
        ax.set_xlim(1, op_counts[-1])
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {os.path.abspath(save_path)}")
    plt.close(fig)


def _save_type_growth_plot(profiler: GraphProfiler, save_path: str, title: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed -- skipping type growth plot")
        return

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
        import importlib
        importlib.reload(plt)

    decomposed = profiler._find_decomposed_parents()
    alive = profiler._build_alive_ranges(decomposed)
    steps, by_type_series, totals = profiler._build_memory_timeline(alive)
    op_counts = [step + 1 for step in steps]

    series_mb = {
        node_type.name: [series.get(node_type, 0) / 1024**2 for series in by_type_series]
        for node_type in NodeType
    }
    total_mb = [value / 1024**2 for value in totals]

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    colors = {
        "PARAM": "#1f77b4",
        "ACT": "#2ca02c",
        "GRAD": "#ff7f0e",
        "OPT": "#8c564b",
        "OTHER": "#9467bd",
    }
    for name in ["PARAM", "ACT", "GRAD", "OPT", "OTHER"]:
        linewidth = 2.5 if name == "OTHER" else 1.8
        alpha = 1.0 if name == "OTHER" else 0.9
        ax.step(
            op_counts,
            series_mb[name],
            where="post",
            linewidth=linewidth,
            color=colors[name],
            alpha=alpha,
            label=name,
        )

    ax.step(
        op_counts,
        total_mb,
        where="post",
        linewidth=2.6,
        color="black",
        linestyle="--",
        label="TOTAL",
    )
    ax.axvline(profiler.sep_idx + 1, color="black", linestyle=":", linewidth=1.1, label="sep")
    ax.axvline(profiler.sep_bw_idx + 1, color="gray", linestyle=":", linewidth=1.1, label="sep_backward")

    ax.set_xlabel("Operations")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"Tensor-type growth over operations{f' -- {title}' if title else ''}")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
    if op_counts:
        ax.set_xlim(1, op_counts[-1])
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {os.path.abspath(save_path)}")
    plt.close(fig)


model_names: List[str] = [
    "Bert",
    "Resnet152",
]

model_batch_sizes: Dict[str, int] = {
    "Bert": 4,
    "Resnet152": 4,
}


class Experiment:
    """Encapsulates model creation, training-step definition, and the
    graph-transformation callback used by compile() to profile a single
    training iteration."""

    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        """Build the model, example inputs, optimizer, and training-step
        closure for the requested architecture."""
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        if self.model_name == "Bert":
            if BertConfig is None or BertForMaskedLM is None:
                raise ImportError(
                    "BERT experiment requires transformers. Install with: pip install transformers"
                )

            vocab_size = 30522
            bsz, seq_len = self.batch_size, 128
            with torch.device(dev):
                bert_cfg = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=4,
                    intermediate_size=1024,
                    max_position_embeddings=512,
                )
                self.model = BertForMaskedLM(bert_cfg)

            input_ids = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            labels = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (input_ids, labels)

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                out = model(input_ids=example_inputs[0], labels=example_inputs[1])
                loss = SEPFunction.apply(out.loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, fused=True, capturable=True)

        elif self.model_name == "Resnet152":
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = resnet152(num_classes=num_classes)

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, fused=True, capturable=True)
            self.train_step = resnet_train_step

    def _time_graph_once_ms(
        self,
        gm: fx.GraphModule,
        args: Any,
    ) -> float:
        """Time one FX graph replay on freshly cloned inputs."""
        bench_args = clone_graph_inputs(args)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gm(*bench_args)
                end.record()
                torch.cuda.synchronize()
                return float(start.elapsed_time(end))

            start_time = time.perf_counter()
            gm(*bench_args)
            return float((time.perf_counter() - start_time) * 1000.0)

    def _measure_latency_comparison(
        self,
        baseline_gm: fx.GraphModule,
        checkpoint_gm: fx.GraphModule,
        args: Any,
        warmup_iters: int = 2,
        measured_iters: int = 5,
    ) -> Dict[str, Any]:
        """Measure baseline/checkpoint graph replay latency in paired trials."""
        measured_iters = max(1, measured_iters)
        warmup_iters = max(0, warmup_iters)

        with torch.no_grad():
            for _ in range(warmup_iters):
                baseline_gm(*clone_graph_inputs(args))
                checkpoint_gm(*clone_graph_inputs(args))

        baseline_samples: List[float] = []
        checkpoint_samples: List[float] = []
        for idx in range(measured_iters):
            if idx % 2 == 0:
                baseline_samples.append(self._time_graph_once_ms(baseline_gm, args))
                checkpoint_samples.append(self._time_graph_once_ms(checkpoint_gm, args))
            else:
                checkpoint_samples.append(self._time_graph_once_ms(checkpoint_gm, args))
                baseline_samples.append(self._time_graph_once_ms(baseline_gm, args))

        baseline_ms = sum(baseline_samples) / len(baseline_samples)
        checkpoint_ms = sum(checkpoint_samples) / len(checkpoint_samples)
        overhead_ms = checkpoint_ms - baseline_ms
        overhead_percent = (checkpoint_ms / baseline_ms - 1.0) * 100.0 if baseline_ms else None
        return {
            "kind": "paired_raw_fx_graph_replay_ms_on_fresh_cloned_inputs",
            "warmup_iters": warmup_iters,
            "measured_iters": measured_iters,
            "baseline_ms": baseline_ms,
            "checkpoint_ms": checkpoint_ms,
            "overhead_ms": overhead_ms,
            "overhead_percent": overhead_percent,
            "baseline_samples_ms": baseline_samples,
            "checkpoint_samples_ms": checkpoint_samples,
        }

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        """Cross-entropy loss, flattened over all sequence positions."""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        """Run one dummy optimizer step so Adam's momentum/variance buffers
        are allocated before tracing (make_fx requires them to exist)."""
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        """Profile the traced graph and save the Phase 1 diagnostics bundle.

        The CUDA2 branch now keeps the active flow on profiling and reporting
        only: no checkpoint plan selection, no graph rewrite, and no policy
        budget knobs.
        """
        print(gm.graph.print_tabular())

        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats()

        plots_dir = _plots_dir()
        os.makedirs(plots_dir, exist_ok=True)
        tag = f"{self.model_name}_bs{self.batch_size}"

        save_path_fw = os.path.join(plots_dir, f"peak_memory_breakdown_fw_{tag}.png")
        save_path_overall = os.path.join(plots_dir, f"peak_memory_breakdown_overall_{tag}.png")
        save_path_timeline = os.path.join(plots_dir, f"memory_vs_opid_{tag}.png")
        save_path_phase = os.path.join(plots_dir, f"memory_by_phase_{tag}.png")
        save_path_components = os.path.join(plots_dir, f"memory_components_{tag}.png")
        save_path_growth = os.path.join(plots_dir, f"memory_growth_{tag}.png")
        save_path_other = os.path.join(plots_dir, f"other_memory_components_{tag}.png")
        save_path_grad = os.path.join(plots_dir, f"gradient_accumulation_{tag}.png")

        try:
            graph_profiler.plot_peak_memory_breakdown(
                title=f"{self.model_name} (bs={self.batch_size})",
                save_path=save_path_fw,
                forward_only=True,
            )
        except Exception as exc:
            print(f"Warning: could not save forward-only plot: {exc}")

        try:
            graph_profiler.plot_peak_memory_breakdown(
                title=f"{self.model_name} (bs={self.batch_size})",
                save_path=save_path_overall,
                forward_only=False,
            )
        except Exception as exc:
            print(f"Warning: could not save overall plot: {exc}")

        try:
            graph_profiler.plot_memory_vs_opid(
                title=f"{self.model_name} (bs={self.batch_size})",
                save_path=save_path_timeline,
            )
        except Exception as exc:
            print(f"Warning: could not save memory-vs-opid plot: {exc}")

        try:
            graph_profiler.plot_phase_memory_summary(
                title=f"{self.model_name} (bs={self.batch_size})",
                save_path=save_path_phase,
            )
        except Exception as exc:
            print(f"Warning: could not save phase-memory plot: {exc}")

        try:
            graph_profiler.plot_memory_components_vs_opid(
                title=f"{self.model_name} (bs={self.batch_size})",
                save_path=save_path_components,
            )
        except Exception as exc:
            print(f"Warning: could not save component-memory plot: {exc}")

        try:
            _save_type_growth_plot(graph_profiler, save_path_growth, f"{self.model_name} (bs={self.batch_size})")
        except Exception as exc:
            print(f"Warning: could not save type-growth plot: {exc}")

        try:
            _save_other_memory_plot(graph_profiler, save_path_other, f"{self.model_name} (bs={self.batch_size})")
        except Exception as exc:
            print(f"Warning: could not save OTHER-memory plot: {exc}")

        try:
            _save_gradient_accumulation_plot(graph_profiler, save_path_grad, f"{self.model_name} (bs={self.batch_size})")
        except Exception as exc:
            print(f"Warning: could not save gradient-accumulation plot: {exc}")

        diagnostics = _build_phase1_diagnostics(graph_profiler)
        diagnostics["model_name"] = self.model_name
        diagnostics["batch_size"] = self.batch_size
        diagnostics["plot_files"] = {
            "forward_peak": os.path.abspath(save_path_fw),
            "overall_peak": os.path.abspath(save_path_overall),
            "memory_vs_opid": os.path.abspath(save_path_timeline),
            "phase_memory": os.path.abspath(save_path_phase),
            "components": os.path.abspath(save_path_components),
            "growth": os.path.abspath(save_path_growth),
            "other_memory_components": os.path.abspath(save_path_other),
            "gradient_accumulation": os.path.abspath(save_path_grad),
        }

        report_path = os.path.join(plots_dir, f"classification_diagnostics_{tag}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Saved diagnostics: {os.path.abspath(report_path)}")

        self.last_diagnostics = diagnostics
        return gm

    def run(self):
        """Execute a single (non-compiled) training step for sanity checking."""
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


if __name__ == "__main__":
    # Usage: python benchmarks.py [model_name] [batch_size]
    # Defaults: Resnet152 with its standard batch size.
    name = sys.argv[1] if len(sys.argv) > 1 else model_names[1]
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else model_batch_sizes[name]
    print(f"Model: {name}, batch_size={bs}\n")
    exp = Experiment(name, bs)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
