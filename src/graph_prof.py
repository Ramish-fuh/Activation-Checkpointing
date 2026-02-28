# =============================================================================
# graph_prof.py — Graph Profiler for Activation Checkpointing Analysis
# =============================================================================
#
# This module implements a profiler that runs a traced PyTorch computation
# graph (fx.GraphModule) node-by-node, performing two kinds of analysis:
#
#   1. STATIC ANALYSIS (done in __init__):
#      - Locates forward/backward pass boundaries using sep/sep_backward markers
#      - Classifies every node as PARAM, ACT (activation), GRAD, or OTHER
#      - Identifies "intermediate activations" — tensors produced in the
#        forward pass that are also consumed in the backward pass. These are
#        the candidates for activation checkpointing.
#      - Records each activation's last forward access and first backward
#        access, which defines its "liveness window."
#
#   2. RUNTIME PROFILING (done in run_node):
#      - Measures per-node GPU execution time using CUDA events
#      - Measures per-node output tensor memory footprint
#      - Computes peak memory analytically by simulating tensor lifetimes
#
# The profiler's outputs feed directly into the activation checkpointing
# algorithm (Phase 2), which decides which activations to recompute vs. retain.
# =============================================================================

import operator
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Tuple

import torch
import torch.fx as fx


class OP(str, Enum):
    """
    Enum of all torch.fx node operation types.
    Used throughout the profiler for node filtering and classification.

    - PLACEHOLDER: input nodes (model params, optimizer states, data batches)
    - CALL_FUNCTION: calls to functions like torch.ops.aten.mm.default
    - CALL_METHOD: calls to tensor methods like .view(), .reshape()
    - CALL_MODULE: calls to nn.Module submodules (rare in traced graphs)
    - GET_ATTR: attribute access on the graph module
    - OUTPUT: the single output node collecting all return values
    """
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    Classification of tensor-producing nodes in the computation graph.

    - PARAM:  Model parameters (weights, biases). These are placeholder nodes
              identified via the optimizer's argument lists.
    - ACT:    Intermediate activations / feature maps. Produced during the
              forward pass AND consumed during the backward pass. These are
              the candidates for activation checkpointing.
    - GRAD:   Gradient tensors. Identified from the optimizer's gradient args
              or from _foreach_addcmul patterns in the backward pass.
    - OTHER:  Everything else (loss computation, optimizer bookkeeping,
              sep/sep_backward markers, scalar ops, etc.).
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


class GraphProfiler(fx.Interpreter):
    """
    Graph profiler that extends fx.Interpreter to run the traced computation
    graph node-by-node, collecting timing and memory statistics.

    Static analysis identifies:
    - Forward/backward pass boundaries (via sep/sep_backward markers)
    - Node types: PARAM, ACT (activation/intermediate), GRAD, OTHER
    - Activation lifetimes: last forward access, first backward access

    Runtime profiling measures:
    - Per-node computation time
    - Per-node output tensor memory
    """

    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        """
        Initialize the profiler: run static analysis on the graph structure,
        then set up storage for runtime profiling data.

        Args:
            module: A traced fx.GraphModule containing the full training step
                    (forward + loss + backward + optimizer).
            garbage_collect_values: If True, the interpreter frees intermediate
                    values once all their consumers have executed.
        """
        super().__init__(module, garbage_collect_values)

        self.device = torch.device("cuda")
        self.device_type = "cuda"

        # ===== STATIC ANALYSIS =====
        # The static analysis below classifies the graph into regions
        # (forward / loss / backward) and labels each node by type
        # (PARAM / ACT / GRAD / OTHER), all without executing anything.

        # Flatten graph nodes into an ordered list, and build a reverse
        # lookup from node -> position index for O(1) ordering queries.
        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_index: Dict[fx.Node, int] = {
            n: i for i, n in enumerate(self.node_list)
        }

        # STEP 1: Find the two sentinel nodes that divide the graph into
        # three regions. These were inserted by the tracer (graph_tracer.py):
        #   sep            — marks the END of the forward pass
        #   sep_backward   — marks the START of the backward pass
        # Between them lies the loss computation.
        self.sep_node: Optional[fx.Node] = None
        self.sep_backward_node: Optional[fx.Node] = None
        for node in self.node_list:
            if node.target == torch.ops.separator.sep.default:
                self.sep_node = node
            elif node.target == torch.ops.separator.sep_backward.default:
                self.sep_backward_node = node
        assert self.sep_node is not None, "sep node not found in graph"
        assert self.sep_backward_node is not None, "sep_backward node not found"

        self.sep_idx = self.node_index[self.sep_node]
        self.sep_bw_idx = self.node_index[self.sep_backward_node]

        # STEP 2: Label every node with its region.
        # "forward"  = all nodes from the start up to and including sep
        #              (this includes placeholders and the forward compute)
        # "loss"     = nodes between sep and sep_backward (loss function)
        # "backward" = sep_backward onwards (autograd backward, optimizer, output)
        self.node_region: Dict[fx.Node, str] = {}
        for i, node in enumerate(self.node_list):
            if i <= self.sep_idx:
                self.node_region[node] = "forward"
            elif i < self.sep_bw_idx:
                self.node_region[node] = "loss"
            else:
                self.node_region[node] = "backward"

        # STEP 3: Identify which placeholder nodes are model parameters,
        # gradients, and optimizer states. Two strategies are tried:
        #   (a) _fused_adam: On CUDA, PyTorch fuses all Adam updates into a
        #       single aten._fused_adam op whose args[0]=params, args[1]=grads,
        #       args[2..4]=optimizer states. Fast and reliable.
        #   (b) _foreach heuristic: On CPU or older PyTorch, Adam decomposes
        #       into _foreach_* ops. We infer params/grads from copy_ targets
        #       and _foreach_addcmul patterns.
        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()
        self.optimizer_state_nodes: Set[fx.Node] = set()

        if not self._identify_via_fused_adam():
            self._identify_via_foreach()

        # STEP 4: Classify every node into one of {PARAM, ACT, GRAD, OTHER}
        # and build the list of "intermediate activations" — forward-pass
        # tensors that cross the forward→backward boundary.
        #
        # For each activation we also record:
        #   last_fw_access  — the last node in the forward/loss region that
        #                     reads this tensor (defines when we COULD free it)
        #   first_bw_access — the first node in the backward region that reads
        #                     this tensor (defines when we MUST have it back)
        # The gap between these two defines the "recomputation window":
        # if we evict the activation after last_fw_access, we must recompute
        # it before first_bw_access.
        self.node_type: Dict[fx.Node, NodeType] = {}
        self.intermediate_nodes: List[fx.Node] = []
        self.last_fw_access: Dict[fx.Node, Optional[fx.Node]] = {}
        self.first_bw_access: Dict[fx.Node, Optional[fx.Node]] = {}

        for node in self.node_list:
            # Already identified as param or grad by Step 3 — label directly
            if node in self.param_nodes:
                self.node_type[node] = NodeType.PARAM
                continue
            if node in self.grad_nodes:
                self.node_type[node] = NodeType.GRAD
                continue

            # Check if this is an intermediate activation:
            #   - Must be in the forward region (not a placeholder/output)
            #   - Must have at least one user in the backward region
            # This means it's a tensor that the backward pass needs for
            # gradient computation, making it an AC candidate.
            if (
                self.node_region[node] == "forward"
                and node.op not in (OP.PLACEHOLDER, OP.OUTPUT)
            ):
                # Collect all users that appear at or after sep_backward
                bw_users = [
                    u for u in node.users
                    if self.node_index[u] >= self.sep_bw_idx
                ]
                if bw_users:
                    self.intermediate_nodes.append(node)
                    self.node_type[node] = NodeType.ACT

                    # Find the LAST node before backward that reads this tensor
                    fw_users = [
                        u for u in node.users
                        if self.node_index[u] < self.sep_bw_idx
                    ]
                    self.last_fw_access[node] = (
                        max(fw_users, key=lambda u: self.node_index[u])
                        if fw_users else None
                    )
                    # Find the FIRST node in backward that reads this tensor
                    self.first_bw_access[node] = min(
                        bw_users, key=lambda u: self.node_index[u]
                    )
                    continue

            # Anything not a param, grad, or forward→backward activation
            self.node_type[node] = NodeType.OTHER

        # ===== RUNTIME PROFILING STORAGE =====
        # These are populated during run_node() when the graph is actually executed.
        #   node_runtimes:   per-node list of elapsed times (ms) across iterations
        #   node_mem_bytes:  per-node output tensor size (bytes), measured once
        #   node_avg_runtime: filled by aggregate_stats() — mean of runtimes
        self.node_runtimes: Dict[str, List[float]] = {
            n.name: [] for n in self.node_list
        }
        self.node_mem_bytes: Dict[str, int] = {}
        self.node_avg_runtime: Dict[str, float] = {}

    # =====================================================================
    # PARAM / GRAD IDENTIFICATION STRATEGIES
    # =====================================================================
    # Two approaches to find which placeholder nodes are parameters vs.
    # gradients vs. optimizer states. The graph structure differs between
    # CUDA (_fused_adam) and CPU (_foreach_*) Adam implementations.
    # =====================================================================

    def _identify_via_fused_adam(self) -> bool:
        """
        CUDA path: find the single aten._fused_adam node and read its args.

        _fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, ...)
        - args[0] = list of parameter nodes
        - args[1] = list of gradient nodes
        - args[2..4] = optimizer state nodes (exp_avg, exp_avg_sq, etc.)

        Returns True if _fused_adam was found, False otherwise.
        """
        for node in self.node_list:
            if node.target == torch.ops.aten._fused_adam.default:
                for p in (node.args[0] or []):
                    if isinstance(p, fx.Node):
                        self.param_nodes.add(p)
                for g in (node.args[1] or []):
                    if isinstance(g, fx.Node):
                        self.grad_nodes.add(g)
                for idx in range(2, min(5, len(node.args))):
                    for s in (node.args[idx] or []):
                        if isinstance(s, fx.Node):
                            self.optimizer_state_nodes.add(s)
                return True
        return False

    def _identify_via_foreach(self):
        """
        CPU / non-fused path: infer params, optimizer states, and gradients
        from graph patterns when _fused_adam is not available.

        Identification logic:
          1. Find all PLACEHOLDER nodes (inputs to the graph).
          2. Find placeholders that are targets of copy_ operations — these
             are being updated in-place by the optimizer.
          3. Among those, if a placeholder also has users in the forward pass,
             it's a MODEL PARAMETER (used in both forward and optimizer).
          4. If it only appears in the optimizer section (no forward users),
             it's an OPTIMIZER STATE (exp_avg, exp_avg_sq, etc.).
          5. GRADIENTS are identified from _foreach_addcmul.Scalar, which
             computes exp_avg_sq += (1-beta2) * grad * grad in Adam.
             args[1] of that op is the list of gradient nodes.
        """
        placeholders = {n for n in self.node_list if n.op == OP.PLACEHOLDER}

        # Find placeholders that are targets of copy_ operations
        copy_targets: Set[fx.Node] = set()
        for node in self.node_list:
            if node.target == torch.ops.aten.copy_.default:
                tgt = node.args[0]
                if isinstance(tgt, fx.Node) and tgt in placeholders:
                    copy_targets.add(tgt)

        for ph in placeholders:
            has_fw_user = any(
                self.node_index.get(u, self.sep_idx) < self.sep_idx
                for u in ph.users
                if u.op != OP.OUTPUT
            )
            if ph in copy_targets:
                if has_fw_user:
                    self.param_nodes.add(ph)
                else:
                    self.optimizer_state_nodes.add(ph)

        # Identify gradient nodes from _foreach_addcmul.Scalar
        # In Adam: exp_avg_sq += (1-beta2) * grad * grad
        # Decomposed as _foreach_addcmul.Scalar(exp_avg_sqs, grads, grads, value)
        # args[1] is the list of gradient nodes
        for node in self.node_list:
            if node.target == torch.ops.aten._foreach_addcmul.Scalar:
                grad_list = node.args[1]
                if isinstance(grad_list, (list, tuple)):
                    for g in grad_list:
                        if isinstance(g, fx.Node) and g not in placeholders:
                            self.grad_nodes.add(g)
                break

    # =====================================================================
    # RUNTIME EXECUTION HOOKS
    # =====================================================================
    # These methods override fx.Interpreter to inject profiling around
    # each node's execution. The interpreter calls run_node() for every
    # node in topological order.
    # =====================================================================

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Execute the full graph with profiling. Delegates to the parent
        Interpreter.run(), which iterates over nodes calling run_node()."""
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: fx.Node) -> Any:
        """
        Execute a single graph node with CUDA timing and memory measurement.

        Timing: Uses paired CUDA events around the node execution to measure
        GPU time in milliseconds (avoids CPU/GPU sync overhead of wall-clock).

        Memory: On the first profiling iteration, records the total byte size
        of the node's output tensor(s). Subsequent iterations skip this since
        tensor sizes don't change between iterations.

        Note: In Phase 2 (activation checkpointing), this method will also
        handle swap-in of evicted activations before backward nodes that
        need them.
        """

        # Record a CUDA event before execution
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        # Actually execute the node (calls the target function with args)
        result = super().run_node(n)

        # Record a CUDA event after execution, synchronize, compute elapsed
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)

        # Append this iteration's timing to the per-node list
        self.node_runtimes[n.name].append(elapsed_ms)

        # Measure output tensor memory once (sizes are constant across iters)
        if n.name not in self.node_mem_bytes:
            self.node_mem_bytes[n.name] = self._tensor_memory(result)

        return result

    # =====================================================================
    # STATS LIFECYCLE
    # =====================================================================
    # Typical workflow: reset → run N warmup iters → reset → run M iters → aggregate → print
    # =====================================================================

    def reset_stats(self) -> None:
        """Clear all collected runtime data. Call before starting a fresh
        measurement window (e.g., after warmup iterations)."""
        for name in self.node_runtimes:
            self.node_runtimes[name] = []
        self.node_avg_runtime = {}

    def aggregate_stats(self) -> None:
        """Compute mean runtime per node from the collected samples.
        Call this after all profiling iterations are done."""
        for name, times in self.node_runtimes.items():
            self.node_avg_runtime[name] = (
                sum(times) / len(times) if times else 0.0
            )

    def print_stats(self) -> None:
        """
        Print a comprehensive summary of profiling results to stdout.

        Sections printed:
          1. Node Classification — counts of PARAM/ACT/GRAD/OTHER nodes
          2. Intermediate Activations — each activation's memory, last forward
             use, and first backward use
          3. Per-Node Profiling — type, region, avg time, and memory for every
             non-placeholder, non-output node
          4. Memory Summary — peak memory breakdown by NodeType
        """
        sep_line = "=" * 90
        print(f"\n{sep_line}")
        print("GRAPH PROFILING RESULTS")
        print(sep_line)

        # --- Classification summary ---
        counts: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        for nt in self.node_type.values():
            counts[nt] += 1
        print("\n--- Node Classification ---")
        for nt, c in counts.items():
            print(f"  {nt.name:8s}: {c} nodes")

        # --- Intermediate activations ---
        print(f"\n--- Intermediate Activations ({len(self.intermediate_nodes)}) ---")
        hdr = (
            f"  {'Name':30s} | {'Memory':>10s} | "
            f"{'Last FW Use':20s} | {'First BW Use':20s}"
        )
        print(hdr)
        print(f"  {'-' * 30}-+-{'-' * 10}-+-{'-' * 20}-+-{'-' * 20}")
        for act in self.intermediate_nodes:
            mem = self.node_mem_bytes.get(act.name, 0)
            lfw = self.last_fw_access.get(act)
            fbw = self.first_bw_access.get(act)
            print(
                f"  {act.name:30s} | {self._fmt_bytes(mem):>10s} | "
                f"{(lfw.name if lfw else 'N/A'):20s} | "
                f"{(fbw.name if fbw else 'N/A'):20s}"
            )

        # --- Per-node profiling (skip placeholders and output) ---
        print(f"\n--- Per-Node Profiling ---")
        hdr2 = (
            f"  {'Name':30s} | {'Type':6s} | {'Region':8s} | "
            f"{'Time(ms)':>10s} | {'Memory':>10s}"
        )
        print(hdr2)
        print(
            f"  {'-' * 30}-+-{'-' * 6}-+-{'-' * 8}-+-"
            f"{'-' * 10}-+-{'-' * 10}"
        )
        for node in self.node_list:
            if node.op in (OP.PLACEHOLDER, OP.OUTPUT):
                continue
            nt = self.node_type.get(node, NodeType.OTHER).name
            region = self.node_region.get(node, "?")
            avg_t = self.node_avg_runtime.get(node.name, 0.0)
            mem = self.node_mem_bytes.get(node.name, 0)
            print(
                f"  {node.name:30s} | {nt:6s} | {region:8s} | "
                f"{avg_t:10.4f} | {self._fmt_bytes(mem):>10s}"
            )

        # --- Memory summary ---
        peak_mem, breakdown = self.compute_peak_memory()
        print(f"\n--- Memory Summary ---")
        for nt in [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER]:
            print(f"  {nt.name:8s}: {self._fmt_bytes(breakdown.get(nt, 0))}")
        print(f"  {'PEAK':8s}: {self._fmt_bytes(peak_mem)}")
        print(sep_line + "\n")

    # =====================================================================
    # PEAK MEMORY ANALYSIS
    # =====================================================================

    def compute_peak_memory(self) -> Tuple[int, Dict[NodeType, int]]:
        """
        Analytically compute peak memory by simulating tensor lifetimes.

        How it works:
          1. For each tensor-producing node, determine its liveness window:
             born = index when the node executes (tensor is allocated)
             dies = index of its last consumer (tensor can be freed after)
             Special case: PARAMs and GRADs persist for the entire iteration.

          2. Handle "decomposed parents": some ops (like _fused_adam) return
             tuples. Their memory is actually in the getitem children, so we
             skip the parent and count each child individually.

          3. Sweep through all steps, summing bytes of all alive tensors at
             each step. Track the maximum and which NodeTypes contribute.

        Returns:
            (peak_total_bytes, {NodeType: bytes_at_peak_step})
        """
        # Some operations (e.g., _fused_adam) return tuples. All their users
        # are operator.getitem extracting individual elements. For accurate
        # per-type memory attribution, we skip the parent node and instead
        # count memory at each getitem child (which has its own NodeType).
        decomposed_parents: Set[fx.Node] = set()
        for node in self.node_list:
            if (
                node.users
                and all(u.target is operator.getitem for u in node.users)
            ):
                decomposed_parents.add(node)

        # Build alive intervals for every tensor-producing node.
        # born = step when the tensor is first produced
        # dies = step of the tensor's last consumer
        # PARAMs and GRADs are special: they persist for the entire iteration
        # (allocated at step 0, freed after the last step).
        last_step = len(self.node_list) - 1
        alive_range: Dict[fx.Node, Tuple[int, int]] = {}
        for node in self.node_list:
            mem = self.node_mem_bytes.get(node.name, 0)
            if mem == 0:
                continue
            # Skip decomposed parents — memory attributed to children
            if node in decomposed_parents:
                continue
            # Skip getitem whose parent is NOT decomposed (rare edge case)
            if node.target is operator.getitem:
                parent = node.args[0]
                if not isinstance(parent, fx.Node) or parent not in decomposed_parents:
                    continue
            nt = self.node_type.get(node, NodeType.OTHER)
            # Params and grads are persistent allocations (born=0, dies=last)
            if nt in (NodeType.PARAM, NodeType.GRAD):
                born = 0
                dies = last_step
            else:
                born = self.node_index[node]
                dies = max(
                    (self.node_index[u] for u in node.users),
                    default=born,
                )
            alive_range[node] = (born, dies)

        # Sweep: at each step, sum memory of all tensors alive at that step.
        # Track the maximum across all steps and the per-type breakdown at peak.
        peak_mem = 0
        peak_breakdown: Dict[NodeType, int] = {}

        for step in range(len(self.node_list)):
            current = 0
            by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
            for node, (born, dies) in alive_range.items():
                if born <= step <= dies:
                    sz = self.node_mem_bytes[node.name]
                    current += sz
                    nt = self.node_type.get(node, NodeType.OTHER)
                    by_type[nt] += sz
            if current > peak_mem:
                peak_mem = current
                peak_breakdown = dict(by_type)

        return peak_mem, peak_breakdown

    # =====================================================================
    # VISUALIZATION
    # =====================================================================

    def plot_peak_memory_breakdown(self, title: str = "", save_path: Optional[str] = None) -> None:
        """
        Generate a bar chart + pie chart of peak memory breakdown by NodeType.

        The bar chart shows absolute MB per category (PARAM, ACT, GRAD, OTHER).
        The pie chart shows the percentage breakdown at the peak memory point.
        Both are saved to save_path as a PNG file (not displayed interactively,
        since this runs on Colab or headless servers with the Agg backend).

        Args:
            title: Optional label appended to the chart title (e.g., model name).
            save_path: File path to save the PNG. If None, a warning is printed.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed, skipping plot")
            return

        # Force non-interactive backend
        backend = matplotlib.get_backend()
        if backend != 'agg':
            matplotlib.use('Agg', force=True)
            import importlib
            importlib.reload(plt)

        peak_mem, breakdown = self.compute_peak_memory()

        categories = [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OTHER]
        labels = [nt.name for nt in categories]
        sizes_mb = [breakdown.get(nt, 0) / 1024 ** 2 for nt in categories]
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        bars = ax1.bar(labels, sizes_mb, color=colors, edgecolor='black')
        for bar, val in zip(bars, sizes_mb):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        ax1.set_ylabel('Memory (MB)')
        ax1.set_title(f'Peak Memory Breakdown{" — " + title if title else ""}')
        ax1.grid(axis='y', alpha=0.3)

        # Pie chart
        nonzero = [(l, s, c) for l, s, c in zip(labels, sizes_mb, colors) if s > 0]
        if nonzero:
            pie_labels, pie_sizes, pie_colors = zip(*nonzero)
            ax2.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 10})
            ax2.set_title(f'Peak: {peak_mem / 1024 ** 2:.1f} MB')

        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {os.path.abspath(save_path)}")
        else:
            print("WARNING: no save_path provided, plot not saved")
        plt.close(fig)

    # =====================================================================
    # HELPER UTILITIES
    # =====================================================================

    @staticmethod
    def _tensor_memory(val: Any) -> int:
        """
        Recursively compute total memory (bytes) of a tensor or nested
        collection of tensors. Returns 0 for non-tensor values.

        Calculation: num_elements * bytes_per_element
        (e.g., a float32 tensor of shape [64, 512] = 64*512*4 = 131072 bytes)
        """
        if isinstance(val, torch.Tensor):
            return val.nelement() * val.element_size()
        if isinstance(val, (tuple, list)):
            return sum(GraphProfiler._tensor_memory(v) for v in val)
        return 0

    @staticmethod
    def _fmt_bytes(b: int) -> str:
        """Format a byte count into a human-readable string (B/KB/MB/GB)."""
        if b < 1024:
            return f"{b} B"
        if b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        if b < 1024 ** 3:
            return f"{b / 1024 ** 2:.1f} MB"
        return f"{b / 1024 ** 3:.2f} GB"
