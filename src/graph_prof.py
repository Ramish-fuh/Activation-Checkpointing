# =============================================================================
# graph_prof.py — Graph Profiler for Activation Checkpointing
# =============================================================================
#
# Extends fx.Interpreter to execute a traced PyTorch computation graph
# node-by-node, performing:
#
#   1. STATIC ANALYSIS  (in __init__, via private helpers)
#      - Locate forward / loss / backward region boundaries
#      - Classify nodes as PARAM, ACT (activation), GRAD, or OTHER
#      - Identify intermediate activations and their liveness windows
#
#   2. RUNTIME PROFILING  (in run_node)
#      - Per-node GPU execution time via CUDA events
#      - Per-node output tensor memory
#      - Analytical peak memory via tensor-lifetime simulation
#
# Outputs feed into the activation checkpointing algorithm (Phase 2).
# =============================================================================

from __future__ import annotations

import operator
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx


# ─────────────────────────────────────────────────────────────────────────────
# Enums  (unchanged from starter code)
# ─────────────────────────────────────────────────────────────────────────────

class OP(str, Enum):
    """torch.fx node operation types used for filtering and classification."""

    CALL_FUNCTION = "call_function"
    CALL_MODULE   = "call_module"
    CALL_METHOD   = "call_method"
    GET_ATTR      = "get_attr"
    OUTPUT        = "output"
    PLACEHOLDER   = "placeholder"


class NodeType(Enum):
    """Semantic label for every node in the traced graph.

    PARAM -- model weight / bias  (identified via optimizer args)
    ACT   -- intermediate activation crossing the fwd -> bwd boundary
    GRAD  -- gradient tensor      (identified via optimizer args)
    OTHER -- loss ops, optimizer bookkeeping, markers, scalars, etc.
    """

    PARAM = 0
    ACT   = 1
    GRAD  = 2
    OTHER = 3


# ─────────────────────────────────────────────────────────────────────────────
# Graph Profiler
# ─────────────────────────────────────────────────────────────────────────────

class GraphProfiler(fx.Interpreter):
    """Profile a traced training-step graph for timing, memory, and
    activation-lifetime data required by the AC algorithm.

    Inherits ``fx.Interpreter`` so we can intercept every node execution
    in ``run_node`` and wrap it with CUDA timing + memory probes.
    """

    # ==================================================================
    # __init__  (matches original skeleton)
    # ==================================================================

    def __init__(
        self,
        module: fx.GraphModule,
        garbage_collect_values: bool = True,
    ) -> None:
        super().__init__(module, garbage_collect_values)

        # -- static analysis (order matters) --
        self._build_node_index()
        self._find_region_boundaries()
        self._label_regions()
        self._identify_params_and_grads()
        self._classify_nodes()

        # -- runtime profiling storage --
        self._init_runtime_storage()

    # ==================================================================
    # run  (matches original skeleton)
    # ==================================================================

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Execute the full graph.  Delegates to ``Interpreter.run``,
        which iterates over nodes calling ``run_node``."""
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    # ==================================================================
    # run_node  (matches original skeleton)
    # ==================================================================

    def run_node(self, n: fx.Node) -> Any:
        """Execute one node, recording CUDA timing and memory.

        Timing -- paired CUDA events avoid CPU/GPU sync overhead.
        Memory -- measured once (tensor sizes are iteration-constant).

        In Phase 2, this method will also handle swapping activations
        back to GPU before backward nodes that consume them.
        """
        start_evt, end_evt = self._create_cuda_events()
        start_evt.record()

        result = super().run_node(n)

        end_evt.record()
        torch.cuda.synchronize()
        self.node_runtimes[n.name].append(start_evt.elapsed_time(end_evt))

        if n.name not in self.node_mem_bytes:
            self.node_mem_bytes[n.name] = self._tensor_bytes(result)

        return result

    # ==================================================================
    # aggregate_stats  (matches original skeleton)
    # ==================================================================

    def aggregate_stats(self) -> None:
        """Average per-node runtimes over all profiling iterations.
        Call after the measurement window (not after warmup)."""
        self.node_avg_runtime = {
            name: (sum(ts) / len(ts) if ts else 0.0)
            for name, ts in self.node_runtimes.items()
        }

    # ==================================================================
    # print_stats  (matches original skeleton)
    # ==================================================================

    def print_stats(self) -> None:
        """Print the full profiling report: classification counts,
        activation table, per-node profiling, peak-memory summary."""
        rule = "=" * 90
        print(f"\n{rule}\nGRAPH PROFILING RESULTS\n{rule}")

        self._print_classification_summary()
        self._print_activation_table()
        self._print_per_node_table()
        self._print_memory_summary()

        print(rule + "\n")

    # ==================================================================
    # reset_stats  (matches original skeleton)
    # ==================================================================

    def reset_stats(self) -> None:
        """Clear runtime data so a fresh measurement window can begin.
        Call after warmup iterations before actual profiling."""
        self.node_runtimes    = {n.name: [] for n in self.node_list}
        self.node_avg_runtime = {}

    # ==================================================================
    # PRIVATE HELPERS -- grouped by responsibility
    # ==================================================================

    # ------------------------------------------------------------------
    # Static Analysis  (called once from __init__)
    # ------------------------------------------------------------------

    def _build_node_index(self) -> None:
        """Flatten the graph into an ordered list with O(1) index lookup."""
        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_index: Dict[fx.Node, int] = {
            node: idx for idx, node in enumerate(self.node_list)
        }

    def _find_region_boundaries(self) -> None:
        """Locate the two sentinel ops inserted by the tracer.

        ``sep``           -- marks the END   of the forward pass
        ``sep_backward``  -- marks the START of the backward pass
        Everything between them is the loss computation.
        """
        self.sep_node:          Optional[fx.Node] = None
        self.sep_backward_node: Optional[fx.Node] = None

        for node in self.node_list:
            if node.target is torch.ops.separator.sep.default:
                self.sep_node = node
            elif node.target is torch.ops.separator.sep_backward.default:
                self.sep_backward_node = node

        assert self.sep_node is not None,          "sep marker not found"
        assert self.sep_backward_node is not None,  "sep_backward marker not found"

        self.sep_idx:    int = self.node_index[self.sep_node]
        self.sep_bw_idx: int = self.node_index[self.sep_backward_node]

    def _label_regions(self) -> None:
        """Tag every node: ``forward`` | ``loss`` | ``backward``."""
        self.node_region: Dict[fx.Node, str] = {}
        for idx, node in enumerate(self.node_list):
            if idx <= self.sep_idx:
                self.node_region[node] = "forward"
            elif idx < self.sep_bw_idx:
                self.node_region[node] = "loss"
            else:
                self.node_region[node] = "backward"

    def _init_runtime_storage(self) -> None:
        """Set up empty containers for profiling data per-iteration. 
        These will be populated in ``run_node`` and aggregated in
        ``aggregate_stats
        """
        self.node_runtimes:    Dict[str, List[float]] = {n.name: [] for n in self.node_list}
        self.node_mem_bytes:   Dict[str, int]         = {}
        self.node_avg_runtime: Dict[str, float]       = {}

    # ------------------------------------------------------------------
    # Param / Grad Identification
    # ------------------------------------------------------------------

    def _identify_params_and_grads(self) -> None:
        """Populate ``param_nodes``, ``grad_nodes``, ``optimizer_state_nodes``.

        Fused Adam (CUDA) is required in this setup. Its args directly list
        params, grads, and optimizer states.
        """
        self.param_nodes:           Set[fx.Node] = set()
        self.grad_nodes:            Set[fx.Node] = set()
        self.optimizer_state_nodes: Set[fx.Node] = set()

        if not self._try_fused_adam():
            raise RuntimeError(
                "Expected aten._fused_adam in FX graph, but it was not found."
            )

    def _try_fused_adam(self) -> bool:
        """Read params / grads / states from ``_fused_adam`` args.

        Signature: ``_fused_adam(params, grads, exp_avgs, exp_avg_sqs, ...)``
        Returns True if the op was found.
        """
        for node in self.node_list:
            if node.target is not torch.ops.aten._fused_adam.default:
                continue

            self._collect_fx_nodes(node.args[0], self.param_nodes)
            self._collect_fx_nodes(node.args[1], self.grad_nodes)
            for i in range(2, min(5, len(node.args))):
                self._collect_fx_nodes(node.args[i], self.optimizer_state_nodes)
            return True

        return False

    # ------------------------------------------------------------------
    # Node Classification
    # ------------------------------------------------------------------

    def _classify_nodes(self) -> None:
        """Assign ``NodeType`` to every node; build intermediate-activation list.

        A node is an *intermediate activation* when:
          1. It lives in the forward region
          2. It is not a placeholder or output
          3. It has at least one consumer in the backward region

        For each activation we record ``last_fw_access`` and ``first_bw_access``
        -- the endpoints of its liveness window.
        """
        self.node_type:          Dict[fx.Node, NodeType]          = {}
        self.intermediate_nodes: List[fx.Node]                    = []
        self.last_fw_access:     Dict[fx.Node, Optional[fx.Node]] = {}
        self.first_bw_access:    Dict[fx.Node, Optional[fx.Node]] = {}

        for node in self.node_list:
            if node in self.param_nodes:
                self.node_type[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_type[node] = NodeType.GRAD
            elif self._is_intermediate_activation(node):
                self._register_activation(node)
            else:
                self.node_type[node] = NodeType.OTHER

    def _is_intermediate_activation(self, node: fx.Node) -> bool:
        """True if *node* is a forward compute node consumed by backward."""
        return (
            self.node_region[node] == "forward"
            and node.op not in (OP.PLACEHOLDER, OP.OUTPUT)
            and any(self.node_index[u] >= self.sep_bw_idx for u in node.users)
        )

    def _register_activation(self, node: fx.Node) -> None:
        """Label *node* as ACT and record its liveness endpoints."""
        self.node_type[node] = NodeType.ACT
        self.intermediate_nodes.append(node)

        fw_users = [u for u in node.users if self.node_index[u] <  self.sep_bw_idx]
        bw_users = [u for u in node.users if self.node_index[u] >= self.sep_bw_idx]

        self.last_fw_access[node] = (
            max(fw_users, key=lambda u: self.node_index[u]) if fw_users else None
        )
        self.first_bw_access[node] = (
            min(bw_users, key=lambda u: self.node_index[u]) if bw_users else None
        )

    # ------------------------------------------------------------------
    # print_stats sub-routines
    # ------------------------------------------------------------------

    def _print_classification_summary(self) -> None:
        counts: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        for nt in self.node_type.values():
            counts[nt] += 1
        print("\n--- Node Classification ---")
        for nt in NodeType:
            print(f"  {nt.name:8s}: {counts[nt]} nodes")

    def _print_activation_table(self) -> None:
        n = len(self.intermediate_nodes)
        print(f"\n--- Intermediate Activations ({n}) ---")
        print(f"  {'Name':30s} | {'Memory':>10s} | {'Last FW Use':20s} | {'First BW Use':20s}")
        print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*20}-+-{'-'*20}")
        for act in self.intermediate_nodes:
            mem = self._fmt_bytes(self.node_mem_bytes.get(act.name, 0))
            lfw = self.last_fw_access.get(act)
            fbw = self.first_bw_access.get(act)
            print(
                f"  {act.name:30s} | {mem:>10s} | "
                f"{(lfw.name if lfw else 'N/A'):20s} | "
                f"{(fbw.name if fbw else 'N/A'):20s}"
            )

    def _print_per_node_table(self) -> None:
        print("\n--- Per-Node Profiling ---")
        print(
            f"  {'Name':30s} | {'Type':6s} | {'Region':8s} | "
            f"{'Time(ms)':>10s} | {'Memory':>10s}"
        )
        print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
        for node in self.node_list:
            if node.op in (OP.PLACEHOLDER, OP.OUTPUT):
                continue
            nt     = self.node_type.get(node, NodeType.OTHER).name
            region = self.node_region.get(node, "?")
            avg_t  = self.node_avg_runtime.get(node.name, 0.0)
            mem    = self._fmt_bytes(self.node_mem_bytes.get(node.name, 0))
            print(f"  {node.name:30s} | {nt:6s} | {region:8s} | {avg_t:10.4f} | {mem:>10s}")

    def _print_memory_summary(self) -> None:
        peak, breakdown = self.compute_peak_memory()
        print("\n--- Memory Summary ---")
        for nt in (NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER):
            print(f"  {nt.name:8s}: {self._fmt_bytes(breakdown.get(nt, 0))}")
        print(f"  {'PEAK':8s}: {self._fmt_bytes(peak)}")

    # ------------------------------------------------------------------
    # Peak-memory analysis
    # ------------------------------------------------------------------

    def compute_peak_memory(self) -> Tuple[int, Dict[NodeType, int]]:
        """Simulate tensor lifetimes; return ``(peak_bytes, {type: bytes})``.

        1. Build alive window ``[born, dies]`` for each tensor node.
        2. Sweep every step summing alive bytes; track the maximum.
        """
        decomposed = self._find_decomposed_parents()
        alive      = self._build_alive_ranges(decomposed)
        return self._sweep_for_peak(alive)

    def _find_decomposed_parents(self) -> Set[fx.Node]:
        """Nodes whose *only* consumers are ``getitem`` (e.g. ``_fused_adam``
        returning a tuple).  Memory is attributed to children instead."""
        return {
            n for n in self.node_list
            if n.users and all(u.target is operator.getitem for u in n.users)
        }

    def _build_alive_ranges(
        self,
        decomposed: Set[fx.Node],
    ) -> Dict[fx.Node, Tuple[int, int]]:
        """``{node: (born, dies)}`` for every tensor-producing node.

        PARAMs / GRADs persist the full iteration.  Others: born = production,
        dies = last consumer.  Decomposed parents are skipped.
        """
        last = len(self.node_list) - 1
        ranges: Dict[fx.Node, Tuple[int, int]] = {}

        for node in self.node_list:
            mem = self.node_mem_bytes.get(node.name, 0)
            if mem == 0 or node in decomposed:
                continue

            # Skip getitem whose parent is NOT decomposed (rare edge case)
            if node.target is operator.getitem:
                parent = node.args[0]
                if not isinstance(parent, fx.Node) or parent not in decomposed:
                    continue

            nt = self.node_type.get(node, NodeType.OTHER)
            if nt in (NodeType.PARAM, NodeType.GRAD):
                ranges[node] = (0, last)
            else:
                born = self.node_index[node]
                dies = max((self.node_index[u] for u in node.users), default=born)
                ranges[node] = (born, dies)

        return ranges

    def _sweep_for_peak(
        self,
        alive: Dict[fx.Node, Tuple[int, int]],
    ) -> Tuple[int, Dict[NodeType, int]]:
        """Walk every step; return ``(peak_bytes, per-type breakdown at peak)``."""
        peak_bytes = 0
        peak_bd:   Dict[NodeType, int] = {}

        for step in range(len(self.node_list)):
            total   = 0
            by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}

            for node, (born, dies) in alive.items():
                if born <= step <= dies:
                    sz = self.node_mem_bytes[node.name]
                    total += sz
                    by_type[self.node_type.get(node, NodeType.OTHER)] += sz

            if total > peak_bytes:
                peak_bytes = total
                peak_bd    = by_type

        return peak_bytes, peak_bd

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_peak_memory_breakdown(
        self,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> None:
        """Save a bar + pie chart of peak memory by NodeType."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed -- skipping plot")
            return

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
            import importlib
            importlib.reload(plt)

        peak, breakdown = self.compute_peak_memory()

        categories = [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER]
        labels     = [nt.name for nt in categories]
        sizes_mb   = [breakdown.get(nt, 0) / 1024**2 for nt in categories]
        colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart -- absolute MB per type
        bars = ax_bar.bar(labels, sizes_mb, color=colors, edgecolor="black")
        for bar, val in zip(bars, sizes_mb):
            if val > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=10,
                )
        ax_bar.set_ylabel("Memory (MB)")
        ax_bar.set_title(f"Peak Memory Breakdown{' -- ' + title if title else ''}")
        ax_bar.grid(axis="y", alpha=0.3)

        # Pie chart -- percentage breakdown
        nonzero = [(l, s, c) for l, s, c in zip(labels, sizes_mb, colors) if s > 0]
        if nonzero:
            pie_labels, pie_sizes, pie_colors = zip(*nonzero)
            ax_pie.pie(
                pie_sizes, labels=pie_labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
            )
            ax_pie.set_title(f"Peak: {peak / 1024**2:.1f} MB")

        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {os.path.abspath(save_path)}")
        else:
            print("WARNING: no save_path -- plot not saved")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_cuda_events() -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create a pair of CUDA events for GPU timing."""
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    @staticmethod
    def _collect_fx_nodes(items: Any, target: Set[fx.Node]) -> None:
        """Add every ``fx.Node`` found in *items* (list/tuple) to *target*."""
        for item in (items or []):
            if isinstance(item, fx.Node):
                target.add(item)

    @staticmethod
    def _tensor_bytes(val: Any) -> int:
        """Total bytes consumed by a tensor or nested collection of tensors."""
        if isinstance(val, torch.Tensor):
            return val.nelement() * val.element_size()
        if isinstance(val, (tuple, list)):
            return sum(GraphProfiler._tensor_bytes(v) for v in val)
        return 0

    @staticmethod
    def _fmt_bytes(b: int) -> str:
        """Human-readable byte string (B / KB / MB / GB)."""
        for unit, threshold in [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]:
            if b >= threshold:
                return f"{b / threshold:.1f} {unit}"
        return f"{b} B"
