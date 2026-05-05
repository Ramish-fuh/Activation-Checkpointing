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
    OPT   -- optimizer state tensor (e.g., Adam exp_avg / exp_avg_sq)
    OTHER -- loss ops, markers, scalars, and uncategorized tensors
    """

    PARAM = 0
    ACT   = 1
    GRAD  = 2
    OPT   = 3
    OTHER = 4


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
            if self._is_alias_node(n):
                self.node_mem_bytes[n.name] = 0
            else:
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
        print("DEBUG_VERSION: graph_prof_debug_2026_04_11_v1")

        self._print_classification_summary()
        self._print_boundary_debug()
        self._print_activation_debug()
        self._print_activation_table()
        self._print_per_node_table()
        self._print_memory_summary()
        self._print_memory_peak_debug()

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
        """Capture the FX graph's existing topological order into a list and index map.

        Does not sort — fx.Graph.nodes is already in topological order.
        node_index[node] gives each node an integer position, which every
        later pass uses to answer positional questions:
          - is this node before or after sep_backward?
          - is a consumer in the forward or backward region?
          - when is a tensor born and when is its last use?
          - give all nodes numbered in topological order.
        """
        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_index: Dict[fx.Node, int] = {
            node: idx for idx, node in enumerate(self.node_list)
        }

    def _find_region_boundaries(self) -> None:
        """Locate the two sentinel ops inserted by the tracer there are only two in the whole graph.

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
        """Tag every node as 'forward', 'loss', or 'backward'.
        main aim is to provide o(1) lookup for later passes that need to know which region a node belongs to.

        Uses the two separator indices as fences:
          idx <= sep_idx          -> forward
          sep_idx < idx < sep_bw_idx  -> loss
          idx >= sep_bw_idx       -> backward
        """
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

            self._collect_fx_nodes(node.args[0], self.param_nodes) # _collect_fx_nodes is a helper that recursively walks nested struvtures
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

        A node is a saved intermediate activation when:
          1. It lives in the forward region
          2. It is not a placeholder or output
          3. It has at least one direct or alias-only consumer in the backward region

        For each activation we record ``last_fw_access`` and ``first_bw_access``
        -- the endpoints of its liveness window.
        """
        self.node_type:          Dict[fx.Node, NodeType]          = {}
        self.intermediate_nodes: List[fx.Node]                    = []
        self.last_fw_access:     Dict[fx.Node, Optional[fx.Node]] = {}
        self.first_bw_access:    Dict[fx.Node, Optional[fx.Node]] = {}
        self.last_bw_access:     Dict[fx.Node, Optional[fx.Node]] = {}

        for node in self.node_list:
            if node in self.param_nodes:
                self.node_type[node] = NodeType.PARAM
            elif node in self.grad_nodes:
                self.node_type[node] = NodeType.GRAD
            elif node in self.optimizer_state_nodes:
                self.node_type[node] = NodeType.OPT
            elif self._is_forward_candidate(node) and self._has_backward_reachability(node):
                self._register_activation(node)
            else:
                self.node_type[node] = NodeType.OTHER

    def _is_forward_candidate(self, node: fx.Node) -> bool:
        """True if *node* is a non-I/O forward intermediate tensor candidate."""
        return (
            self.node_index[node] < self.sep_idx
            and node.op not in (OP.PLACEHOLDER, OP.OUTPUT)
            and node.target is not torch.ops.separator.sep.default
            and not self._is_alias_node(node)
        )

    def _is_alias_node(self, node: fx.Node) -> bool:
        """True for view-like ops that share storage with another tensor.

        These nodes can carry saved-tensor edges into backward, but their
        output does not own a fresh activation buffer. We follow them when
        finding backward consumers, then count memory on the real producer.
        """
        target_name = str(node.target)
        alias_markers = (
            "aten.view.",
            "aten._unsafe_view.",
            "aten.t.",
            "aten.transpose.",
            "aten.permute.",
            "aten.expand.",
            "aten.squeeze.",
            "aten.unsqueeze.",
            "aten.slice.",
            "aten.select.",
            "aten.as_strided.",
            "aten.detach.",
            "aten.alias.",
        )
        if any(marker in target_name for marker in alias_markers):
            return True

        if node.op == OP.CALL_METHOD and node.target in {
            "view",
            "t",
            "transpose",
            "permute",
            "expand",
            "squeeze",
            "unsqueeze",
            "slice",
            "select",
            "detach",
        }:
            return True

        return False

    def _activation_use_sets(self, node: fx.Node) -> Tuple[Set[fx.Node], Set[fx.Node]]:
        """Return direct forward users and alias-only backward consumers.

        Backward consumers may be reached through view-like alias nodes, because
        autograd can save a view while the actual storage belongs to the view's
        producer. We deliberately do not walk through real forward compute ops;
        doing so would make most activations appear to be first used by the loss
        backward op instead of by their saved-tensor consumer.
        """
        forward_users: Set[fx.Node] = set()
        backward_consumers: Set[fx.Node] = set()
        seen_alias_users: Set[fx.Node] = set()

        def visit_user(user: fx.Node) -> None:
            if user in seen_alias_users:
                return
            seen_alias_users.add(user)
            u_idx = self.node_index[user]
            if u_idx < self.sep_idx:
                forward_users.add(user)
                if self._is_alias_node(user):
                    for alias_user in user.users:
                        visit_user(alias_user)
            elif u_idx >= self.sep_bw_idx and user.target is not torch.ops.separator.sep_backward.default:
                backward_consumers.add(user)

        for user in node.users:
            visit_user(user)

        return forward_users, backward_consumers

    def _has_backward_reachability(self, node: fx.Node) -> bool:
        """True if any direct consumer of *node* is a real backward op.

        Excludes the ``sep_backward`` marker itself, which is only a boundary
        sentinel and should not count as a true activation use.
        """
        _, bw_users = self._activation_use_sets(node)
        return len(bw_users) > 0

    def _register_activation(self, node: fx.Node) -> None:
        """Label *node* as ACT and record where its lifetime crosses regions.

        Saves the last forward consumer and first backward consumer,
        which are the key endpoints for activation liveness.
        """
        self.node_type[node] = NodeType.ACT
        self.intermediate_nodes.append(node)

        fw_users, bw_users_set = self._activation_use_sets(node)
        bw_users = list(bw_users_set)

        self.last_fw_access[node] = (
            max(fw_users, key=lambda u: self.node_index[u]) if fw_users else None
        )
        self.first_bw_access[node] = (
            min(bw_users, key=lambda u: self.node_index[u]) if bw_users else None
        )
        self.last_bw_access[node] = (
            max(bw_users, key=lambda u: self.node_index[u]) if bw_users else None
        )

    # ------------------------------------------------------------------
    # print_stats sub-routines
    # ------------------------------------------------------------------

    def _print_classification_summary(self) -> None:
        """Print a count of nodes per NodeType (PARAM / ACT / GRAD / OTHER)."""
        counts: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        for nt in self.node_type.values():
            counts[nt] += 1
        fw_candidates = sum(1 for n in self.node_list if self._is_forward_candidate(n))
        checkpointable = sum(1 for n in self.intermediate_nodes if self.first_bw_access.get(n) is not None)
        print("\n--- Node Classification ---")
        for nt in NodeType:
            print(f"  {nt.name:8s}: {counts[nt]} nodes")
        print(f"  {'FW candidates':14s}: {fw_candidates} nodes")
        print(f"  {'ACT(w/ BW use)':14s}: {checkpointable} nodes")

    def _print_boundary_debug(self) -> None:
        """Print boundary marker diagnostics for forward/loss/backward split."""
        print("\n--- Boundary Debug ---")
        print(f"  sep idx={self.sep_idx}, name={self.sep_node.name}")
        print(f"  sep_backward idx={self.sep_bw_idx}, name={self.sep_backward_node.name}")

        def _node_desc(idx: int) -> str:
            if idx < 0 or idx >= len(self.node_list):
                return "<out-of-range>"
            n = self.node_list[idx]
            return f"{idx}:{n.name} ({n.op})"

        print("  Around sep:")
        for i in range(self.sep_idx - 2, self.sep_idx + 3):
            print(f"    {_node_desc(i)}")

        print("  Around sep_backward:")
        for i in range(self.sep_bw_idx - 2, self.sep_bw_idx + 3):
            print(f"    {_node_desc(i)}")

    def _print_activation_debug(self) -> None:
        """Print ACT integrity diagnostics and suspicious cases."""
        print("\n--- Activation Debug ---")
        act_nodes = [n for n in self.node_list if self.node_type.get(n) is NodeType.ACT]
        no_fbw = [n for n in act_nodes if self.first_bw_access.get(n) is None]
        sep_fbw = [
            n for n in act_nodes
            if (self.first_bw_access.get(n) is not None
                and self.first_bw_access[n].target is torch.ops.separator.sep_backward.default)
        ]

        print(f"  total ACT nodes: {len(act_nodes)}")
        print(f"  ACT with no first_bw_access: {len(no_fbw)}")
        print(f"  ACT with first_bw_access == sep_backward: {len(sep_fbw)}")

        if no_fbw:
            print("  sample ACT with no first_bw_access:")
            for n in no_fbw[:20]:
                print(f"    - {n.name}")

        # Show a few FW candidates that failed backward reachability.
        non_act_fw = [
            n for n in self.node_list
            if self._is_forward_candidate(n) and self.node_type.get(n) is not NodeType.ACT
        ]
        print(f"  FW candidates not labeled ACT: {len(non_act_fw)}")
        if non_act_fw:
            print("  sample FW candidates not ACT:")
            for n in non_act_fw[:20]:
                print(f"    - {n.name} ({n.op})")

    def _print_activation_table(self) -> None:
        """Print each intermediate activation with its memory size and liveness endpoints."""
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
        """Print type, region, average runtime, and memory for every non-I/O node."""
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
        """Print simulated peak memory broken down by NodeType."""
        peak, breakdown = self.compute_peak_memory()
        print("\n--- Memory Summary ---")
        for nt in (NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPT, NodeType.OTHER):
            print(f"  {nt.name:8s}: {self._fmt_bytes(breakdown.get(nt, 0))}")
        print(f"  {'PEAK':8s}: {self._fmt_bytes(peak)}")

    def _print_memory_peak_debug(self) -> None:
        """Print detailed diagnostics for peak and ACT-specific memory behavior."""
        print("\n--- Memory Peak Debug ---")
        decomposed = self._find_decomposed_parents()
        alive = self._build_alive_ranges(decomposed)

        peak_step = -1
        peak_total = 0
        peak_by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}

        max_act_step = -1
        max_act_bytes = 0

        for step in range(len(self.node_list)):
            total = 0
            by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
            for node, (born, dies) in alive.items():
                if born <= step <= dies:
                    sz = self.node_mem_bytes[node.name]
                    total += sz
                    by_type[self.node_type.get(node, NodeType.OTHER)] += sz

            if total > peak_total:
                peak_total = total
                peak_step = step
                peak_by_type = by_type

            act_bytes = by_type.get(NodeType.ACT, 0)
            if act_bytes > max_act_bytes:
                max_act_bytes = act_bytes
                max_act_step = step

        peak_region = self.node_region.get(self.node_list[peak_step], "?") if peak_step >= 0 else "?"
        max_act_region = self.node_region.get(self.node_list[max_act_step], "?") if max_act_step >= 0 else "?"

        # Forward-only peak
        fw_peak_step, fw_peak_total, fw_peak_by_type = self._sweep_for_forward_peak(alive)
        fw_peak_region = self.node_region.get(self.node_list[fw_peak_step], "?") if fw_peak_step >= 0 else "?"

        print(f"  peak step (overall): {peak_step}, region: {peak_region}, total: {self._fmt_bytes(peak_total)}")
        print(f"  peak step (forward only): {fw_peak_step}, region: {fw_peak_region}, total: {self._fmt_bytes(fw_peak_total)}")
        print(f"  peak ACT at any step: step {max_act_step}, region: {max_act_region}, ACT={self._fmt_bytes(max_act_bytes)}")
        print(
            "  breakdown at overall peak: "
            f"PARAM={self._fmt_bytes(peak_by_type.get(NodeType.PARAM, 0))}, "
            f"ACT={self._fmt_bytes(peak_by_type.get(NodeType.ACT, 0))}, "
            f"GRAD={self._fmt_bytes(peak_by_type.get(NodeType.GRAD, 0))}, "
            f"OPT={self._fmt_bytes(peak_by_type.get(NodeType.OPT, 0))}, "
            f"OTHER={self._fmt_bytes(peak_by_type.get(NodeType.OTHER, 0))}"
        )
        print(
            "  breakdown at forward peak: "
            f"PARAM={self._fmt_bytes(fw_peak_by_type.get(NodeType.PARAM, 0))}, "
            f"ACT={self._fmt_bytes(fw_peak_by_type.get(NodeType.ACT, 0))}, "
            f"GRAD={self._fmt_bytes(fw_peak_by_type.get(NodeType.GRAD, 0))}, "
            f"OPT={self._fmt_bytes(fw_peak_by_type.get(NodeType.OPT, 0))}, "
            f"OTHER={self._fmt_bytes(fw_peak_by_type.get(NodeType.OTHER, 0))}"
        )

        if fw_peak_step >= 0:
            alive_at_fw_peak: List[Tuple[int, str, str]] = []
            for node, (born, dies) in alive.items():
                if born <= fw_peak_step <= dies:
                    alive_at_fw_peak.append(
                        (
                            self.node_mem_bytes[node.name],
                            node.name,
                            self.node_type.get(node, NodeType.OTHER).name,
                        )
                    )
            alive_at_fw_peak.sort(reverse=True)
            print("  top alive tensors at forward peak (size, type, name):")
            for sz, name, nt in alive_at_fw_peak[:25]:
                print(f"    - {self._fmt_bytes(sz):>10s} | {nt:5s} | {name}")

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
        """Find tuple/container nodes consumed only by ``getitem``; skip them so
        memory is counted on the extracted child tensors, not twice."""
        return {
            n for n in self.node_list
            if n.users and all(u.target is operator.getitem for u in n.users)
        }

    def _collect_recompute_closure(
        self,
        target: fx.Node,
        recompute_nodes: Optional[Set[fx.Node]] = None,
    ) -> Set[fx.Node]:
        """Collect the forward-region nodes needed to recompute *target*.

        This mirrors the dependency walk used by the checkpoint policy and is
        used to mark temporary recompute dependencies as short-lived.
        """
        recompute_nodes = set(recompute_nodes or {target})
        closure: Set[fx.Node] = set()
        visited: Set[fx.Node] = set()

        def is_retained_activation_boundary(node: fx.Node) -> bool:
            return (
                node not in recompute_nodes
                and self.node_type.get(node) is NodeType.ACT
                and self.first_bw_access.get(node) is not None
            )

        def visit(node: fx.Node) -> None:
            if node in visited:
                return
            visited.add(node)

            if self.node_region.get(node) != "forward":
                return

            nt = self.node_type.get(node, NodeType.OTHER)
            if node.op == OP.PLACEHOLDER or nt is NodeType.PARAM:
                closure.add(node)
                return

            if is_retained_activation_boundary(node):
                closure.add(node)
                return

            closure.add(node)
            for inp in node.all_input_nodes:
                visit(inp)

        visit(target)
        return closure

    def _build_alive_ranges(
        self,
        decomposed: Set[fx.Node],
    ) -> Dict[fx.Node, Tuple[int, int]]:
        """Build per-node lifetime windows as ``{node: (born_idx, dies_idx)}``.

        A node is considered alive for every step ``i`` where
        ``born_idx <= i <= dies_idx``.

        Rules:
        - Skip nodes with 0 measured bytes.
        - Skip tuple/container parents listed in ``decomposed``.
        - PARAM, GRAD, and OPT nodes live for the full iteration: ``(0, last_idx)``.
        - Other tensor nodes live from their own index to their last consumer.
        - ``getitem`` nodes are counted only when they extract from a decomposed
            parent (to avoid counting unrelated edge cases).
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
            if nt is NodeType.PARAM:
                ranges[node] = (0, last)
            elif nt is NodeType.GRAD:
                # Gradients materialize when their producer node executes
                # (typically in backward), then persist until step end.
                born = min(max(self.node_index[node], self.sep_bw_idx), last)
                ranges[node] = (born, last)
            elif nt is NodeType.OPT:
                # Optimizer states are treated as backward-resident for peak accounting.
                born = min(self.sep_bw_idx, last)
                ranges[node] = (born, last)
            elif nt is NodeType.ACT:
                born = self.node_index[node]
                lbw = self.last_bw_access.get(node)
                if lbw is None:
                    raise RuntimeError(
                        f"Invariant violation: ACT node {node.name} has no backward use"
                    )
                dies = self.node_index[lbw]
                ranges[node] = (born, dies)
            else:
                born = self.node_index[node]
                dies = max((self.node_index[u] for u in node.users), default=born)
                ranges[node] = (born, dies)

        return ranges

    def _sweep_for_peak(
        self,
        alive: Dict[fx.Node, Tuple[int, int]],
    ) -> Tuple[int, Dict[NodeType, int]]:
        """Sweep the node timeline to find peak alive memory (optimized with events).

        Uses event-based sweep: track birth/death of each node, accumulate on-the-fly.
        Reduces O(n*m) to O((n+m)*log(n+m)) with sorting.
        """
        events: List[Tuple[int, int, fx.Node]] = []  # (step, kind, node): kind=0 for birth, 1 for death
        for node, (born, dies) in alive.items():
            events.append((born, 0, node))  # birth event
            events.append((dies + 1, 1, node))  # death event (exclusive upper bound)

        events.sort()

        peak_bytes = 0
        peak_bd: Dict[NodeType, int] = {}
        current_by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        current_total = 0

        for step, kind, node in events:
            if step < len(self.node_list):
                sz = self.node_mem_bytes[node.name]
                nt = self.node_type.get(node, NodeType.OTHER)
                if kind == 0:  # birth
                    current_total += sz
                    current_by_type[nt] = current_by_type.get(nt, 0) + sz
                else:  # death
                    current_total -= sz
                    current_by_type[nt] = current_by_type.get(nt, 0) - sz

                if current_total > peak_bytes:
                    peak_bytes = current_total
                    peak_bd = dict(current_by_type)

        return peak_bytes, peak_bd

    def _sweep_for_forward_peak(
        self,
        alive: Dict[fx.Node, Tuple[int, int]],
    ) -> Tuple[int, int, Dict[NodeType, int]]:
        """Sweep forward-pass region to find peak memory (optimized with events).

        Uses event-based sweep for efficiency.
        """
        forward_end = min(self.sep_idx + 1, len(self.node_list))

        events: List[Tuple[int, int, fx.Node]] = []
        for node, (born, dies) in alive.items():
            if born < forward_end and dies >= 0:
                events.append((max(0, born), 0, node))  # birth event
                events.append((min(dies + 1, forward_end), 1, node))  # death event

        events.sort()

        peak_step = -1
        peak_bytes = 0
        peak_bd: Dict[NodeType, int] = {}
        current_by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        current_total = 0
        current_step = 0

        for step, kind, node in events:
            # Track the step when this max is reached
            if step < forward_end and current_total > peak_bytes:
                peak_bytes = current_total
                peak_bd = dict(current_by_type)
                peak_step = current_step

            sz = self.node_mem_bytes[node.name]
            nt = self.node_type.get(node, NodeType.OTHER)
            if kind == 0:  # birth
                current_total += sz
                current_by_type[nt] = current_by_type.get(nt, 0) + sz
                current_step = step
            else:  # death
                current_total -= sz
                current_by_type[nt] = current_by_type.get(nt, 0) - sz

        if current_total > peak_bytes and current_step < forward_end:
            peak_bytes = current_total
            peak_bd = dict(current_by_type)
            peak_step = current_step

        return peak_step, peak_bytes, peak_bd

    def _build_alive_ranges_with_checkpoint(
        self,
        decomposed: Set[fx.Node],
        checkpoint_plan: Any,
    ) -> Dict[fx.Node, Tuple[int, int]]:
        """Build alive ranges with checkpoint behavior accounted for.

        Retained activations keep their original liveness.

        Nodes in a recompute closure are treated as temporary:
        - the activation itself stays alive from first backward use through its
          last backward consumer,
        - intermediate recompute dependencies are only alive at the recompute
          event and are released immediately after producing the recomputed node.
        """
        base_alive = self._build_alive_ranges(decomposed)
        adjusted: Dict[fx.Node, Tuple[int, int]] = {}

        retained_nodes = getattr(checkpoint_plan, "retained_nodes", set())
        recompute_acts = getattr(checkpoint_plan, "recompute_nodes", set())
        first_bw_use = getattr(checkpoint_plan, "first_backward_use", {})

        temp_dep_step: Dict[fx.Node, int] = {}
        boundary_input_last_step: Dict[fx.Node, int] = {}
        act_live_ranges: Dict[fx.Node, Tuple[int, int]] = {}

        for act in recompute_acts:
            fbw = first_bw_use.get(act, self.first_bw_access.get(act))
            if fbw is None:
                continue
            fbw_idx = self.node_index[fbw]
            lbw = self.last_bw_access.get(act, fbw)
            lbw_idx = self.node_index[lbw]
            act_live_ranges[act] = (fbw_idx, lbw_idx)

            for dep in self._collect_recompute_closure(act, recompute_acts):
                if dep is act or dep in retained_nodes:
                    continue
                dep_type = self.node_type.get(dep, NodeType.OTHER)
                if dep.op == OP.PLACEHOLDER or dep_type in (NodeType.PARAM, NodeType.GRAD, NodeType.OPT):
                    if dep.op == OP.PLACEHOLDER:
                        boundary_input_last_step[dep] = max(
                            boundary_input_last_step.get(dep, -1),
                            fbw_idx,
                        )
                    continue
                if dep not in temp_dep_step or fbw_idx < temp_dep_step[dep]:
                    temp_dep_step[dep] = fbw_idx

        for node, (born, dies) in base_alive.items():
            nt = self.node_type.get(node, NodeType.OTHER)
            if nt in (NodeType.PARAM, NodeType.GRAD, NodeType.OPT):
                adjusted[node] = (born, dies)
                continue

            if node in boundary_input_last_step:
                adjusted[node] = (born, max(dies, boundary_input_last_step[node]))
                continue

            if node in act_live_ranges:
                adjusted[node] = act_live_ranges[node]
                continue

            if node in temp_dep_step:
                step = temp_dep_step[node]
                adjusted[node] = (step, step)
                continue

            if nt is not NodeType.ACT:
                adjusted[node] = (born, dies)
                continue

            if node in retained_nodes:
                adjusted[node] = (born, dies)
                continue

            # Unretained activations that are not explicit recompute targets are
            # treated as single-use temporaries.
            fbw = self.first_bw_access.get(node)
            if fbw is None:
                continue
            fbw_idx = self.node_index[fbw]
            adjusted[node] = (fbw_idx, fbw_idx)

        return adjusted

    def _build_memory_timeline(
        self,
        alive: Dict[fx.Node, Tuple[int, int]],
    ) -> Tuple[List[int], List[Dict[NodeType, int]], List[int]]:
        """Return per-step memory timeline (optimized with events).
        
        Builds full timeline using event sweep: O((n+m)*log(n+m)) instead of O(n*m).
        """
        steps = list(range(len(self.node_list)))
        
        # Build events for birth/death at each step
        events: List[Tuple[int, int, fx.Node]] = []
        for node, (born, dies) in alive.items():
            events.append((born, 0, node))  # birth at step born
            events.append((dies + 1, 1, node))  # death after step dies (exclusive)
        
        events.sort()
        
        by_type_series: List[Dict[NodeType, int]] = []
        totals: List[int] = []
        
        current_by_type: Dict[NodeType, int] = {nt: 0 for nt in NodeType}
        current_total = 0
        event_idx = 0
        
        for step in steps:
            # Process all events at this step
            while event_idx < len(events) and events[event_idx][0] == step:
                _, kind, node = events[event_idx]
                sz = self.node_mem_bytes[node.name]
                nt = self.node_type.get(node, NodeType.OTHER)
                if kind == 0:  # birth
                    current_total += sz
                    current_by_type[nt] = current_by_type.get(nt, 0) + sz
                else:  # death
                    current_total -= sz
                    current_by_type[nt] = current_by_type.get(nt, 0) - sz
                event_idx += 1
            
            by_type_series.append(dict(current_by_type))
            totals.append(current_total)
        
        return steps, by_type_series, totals

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_peak_memory_breakdown(
        self,
        title: str = "",
        save_path: Optional[str] = None,
        forward_only: bool = True,
        checkpoint_plan: Optional[Any] = None,
    ) -> None:
        """Save a bar + pie chart of peak memory by NodeType.
        
        Args:
            title: Optional title suffix.
            save_path: Where to save the plot file.
            forward_only: If True, plot forward-pass peak; if False, plot overall peak.
            checkpoint_plan: Optional CheckpointPlan to show impact of checkpointing.
                If provided, generates two plots: baseline and with-checkpoint.
        """
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

        decomposed = self._find_decomposed_parents()
        alive = self._build_alive_ranges(decomposed)

        if forward_only:
            fw_peak_step, peak, breakdown = self._sweep_for_forward_peak(alive)
            plot_region = "Forward Pass"
        else:
            peak, breakdown = self.compute_peak_memory()
            plot_region = "Overall"

        # Generate baseline plot
        self._generate_single_plot(
            breakdown, peak, plot_region, title, save_path, "Baseline (Without Checkpoint)"
        )

        # If checkpoint plan provided, generate with-checkpoint plot using the
        # same recomputation-aware lifetime model as timeline/phase plots.
        if checkpoint_plan is not None:
            alive_with_cp = self._build_alive_ranges_with_checkpoint(
                decomposed,
                checkpoint_plan,
            )
            if forward_only:
                fw_peak_step_cp, peak_cp, breakdown_cp = self._sweep_for_forward_peak(alive_with_cp)
            else:
                peak_cp, breakdown_cp = self.compute_peak_memory_filtered(alive_with_cp)
            
            if save_path:
                cp_save_path = save_path.replace(".png", "_with_checkpoint.png")
            else:
                cp_save_path = None
            
            self._generate_single_plot(
                breakdown_cp,
                peak_cp,
                plot_region,
                title,
                cp_save_path,
                "Checkpoint Plan (Modeled)",
            )

    def plot_memory_vs_opid(
        self,
        title: str = "",
        save_path: Optional[str] = None,
        checkpoint_plan: Optional[Any] = None,
    ) -> None:
        """Plot memory-over-time against op id with region markers.

        If ``checkpoint_plan`` is provided, overlays a checkpoint-aware
        timeline where recomputed activations are materialized only at
        first backward consumption and then released immediately.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed -- skipping memory-vs-opid plot")
            return

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
            import importlib
            importlib.reload(plt)

        decomposed = self._find_decomposed_parents()
        # Build the alive ranges but do not plot the total-alive series
        # (user requested removing TOTAL / "total alive" from all plots).
        alive_baseline = self._build_alive_ranges(decomposed)
        steps, _, _ = self._build_memory_timeline(alive_baseline)

        op_counts = [s + 1 for s in steps]
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        # If a checkpoint plan is provided we still prepare an adjusted alive
        # set for other annotations, but deliberately avoid plotting any
        # aggregate "total" series.
        if checkpoint_plan is not None:
            alive_cp = self._build_alive_ranges_with_checkpoint(decomposed, checkpoint_plan)

        ax.axvline(self.sep_idx + 1, color="black", linestyle=":", linewidth=1.3, label="sep")
        ax.axvline(self.sep_bw_idx + 1, color="gray", linestyle=":", linewidth=1.3, label="sep_backward")

        ax.set_xlabel("Operations (count)")
        ax.set_ylabel("Alive Memory (MB)")
        ax.set_title(f"Memory vs Op ID{' -- ' + title if title else ''}")
        ax.grid(alpha=0.3)
        ax.legend()
        if op_counts:
            ax.set_xlim(1, op_counts[-1])
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {os.path.abspath(save_path)}")
        else:
            print("WARNING: no save_path -- plot not saved")
        plt.close(fig)

    def plot_phase_memory_summary(
        self,
        title: str = "",
        save_path: Optional[str] = None,
        checkpoint_plan: Optional[Any] = None,
    ) -> None:
        """Plot peak memory by execution phase (forward/loss/backward)."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed -- skipping phase-memory plot")
            return

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
            import importlib
            importlib.reload(plt)

        def _phase_peaks(totals: List[int]) -> Dict[str, float]:
            fw_end = min(self.sep_idx, len(totals) - 1)
            loss_end = min(self.sep_bw_idx - 1, len(totals) - 1)

            fw_peak = max(totals[: fw_end + 1], default=0)
            loss_peak = max(totals[self.sep_idx + 1 : loss_end + 1], default=0)
            bw_peak = max(totals[self.sep_bw_idx :], default=0)
            return {
                "Forward": fw_peak / 1024**2,
                "Loss": loss_peak / 1024**2,
                "Backward": bw_peak / 1024**2,
            }

        decomposed = self._find_decomposed_parents()
        alive_baseline = self._build_alive_ranges(decomposed)
        _, _, totals = self._build_memory_timeline(alive_baseline)
        baseline = _phase_peaks(totals)

        phases = ["Forward", "Loss", "Backward"]
        x = list(range(len(phases)))
        width = 0.38

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        base_vals = [baseline[p] for p in phases]
        ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline", color="#1f77b4")

        if checkpoint_plan is not None:
            alive_cp = self._build_alive_ranges_with_checkpoint(decomposed, checkpoint_plan)
            _, _, totals_cp = self._build_memory_timeline(alive_cp)
            cp = _phase_peaks(totals_cp)
            cp_vals = [cp[p] for p in phases]
            ax.bar(
                [i + width / 2 for i in x],
                cp_vals,
                width,
                label="Checkpoint Plan (modeled)",
                color="#d62728",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel("Peak Memory in Phase (MB)")
        ax.set_title(f"Memory by Pass Phase{' -- ' + title if title else ''}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {os.path.abspath(save_path)}")
        else:
            print("WARNING: no save_path -- plot not saved")
        plt.close(fig)

    def plot_memory_components_vs_opid(
        self,
        title: str = "",
        save_path: Optional[str] = None,
        checkpoint_plan: Optional[Any] = None,
    ) -> None:
        """Plot per-component memory against op id.

        Components:
        - weights   -> NodeType.PARAM
        - gradients -> NodeType.GRAD
        - feature maps -> NodeType.ACT
        - other tensors -> NodeType.OTHER

        This view isolates feature-map liveness so forward rise / backward fall
        is visible even when total memory is dominated by other tensor classes.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed -- skipping component-memory plot")
            return

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
            import importlib
            importlib.reload(plt)

        decomposed = self._find_decomposed_parents()
        if checkpoint_plan is None:
            alive = self._build_alive_ranges(decomposed)
        else:
            alive = self._build_alive_ranges_with_checkpoint(decomposed, checkpoint_plan)

        steps, by_type_series, _ = self._build_memory_timeline(alive)
        op_counts = [s + 1 for s in steps]
        weights_mb = [s.get(NodeType.PARAM, 0) / 1024**2 for s in by_type_series]
        grads_mb = [s.get(NodeType.GRAD, 0) / 1024**2 for s in by_type_series]
        feats_mb = [s.get(NodeType.ACT, 0) / 1024**2 for s in by_type_series]
        other_mb = [s.get(NodeType.OTHER, 0) / 1024**2 for s in by_type_series]
        # Match the diagnostic style where weights are shown as a near-horizontal
        # baseline across the full operation axis.
        w_level = max(weights_mb, default=0.0)
        weights_line = [w_level] * len(op_counts)

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(op_counts, weights_line, color="#1f77b4", linewidth=1.8, label="weights")
        ax.step(op_counts, grads_mb, where="post", color="#ff7f0e", linewidth=1.8, label="gradients")
        ax.step(op_counts, feats_mb, where="post", color="#2ca02c", linewidth=1.8, label="feature maps")
        ax.step(op_counts, other_mb, where="post", color="#9467bd", linewidth=1.8, label="other")
        ax.axvline(self.sep_bw_idx + 1, color="black", linestyle="--", linewidth=1.1, label="fw_bw_boundary")

        ax.set_xlabel("operations")
        ax.set_ylabel("Memory (MB)")
        suffix = " (checkpoint plan modeled)" if checkpoint_plan is not None else ""
        ax.set_title(f"Memory Components vs Operations{suffix}{' -- ' + title if title else ''}")
        ax.grid(alpha=0.3)
        ax.legend()
        if op_counts:
            ax.set_xlim(1, op_counts[-1])
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {os.path.abspath(save_path)}")
        else:
            print("WARNING: no save_path -- plot not saved")
        plt.close(fig)

    def compute_peak_memory_filtered(
        self,
        alive: Dict[fx.Node, Tuple[int, int]],
    ) -> Tuple[int, Dict[NodeType, int]]:
        """Compute peak memory from a filtered alive set."""
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

    def _generate_single_plot(
        self,
        breakdown: Dict[NodeType, int],
        peak: int,
        region: str,
        title: str,
        save_path: Optional[str],
        scenario: str,
    ) -> None:
        """Generate a single bar+pie plot for a given scenario."""
        import matplotlib.pyplot as plt

        categories = [NodeType.PARAM, NodeType.ACT, NodeType.GRAD, NodeType.OPT, NodeType.OTHER]
        labels     = [nt.name for nt in categories]
        sizes_bytes = [breakdown.get(nt, 0) for nt in categories]
        sizes_mb   = [size / 1024**2 for size in sizes_bytes]
        colors     = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52"]

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart -- absolute MB per type
        bars = ax_bar.bar(labels, sizes_mb, color=colors, edgecolor="black")
        for bar, val_mb, val_bytes in zip(bars, sizes_mb, sizes_bytes):
            if val_bytes > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    self._fmt_bytes(val_bytes),
                    ha="center", va="bottom", fontsize=10,
                )
        ax_bar.set_ylabel("Memory (MB)")
        ax_bar.set_title(f"Peak Memory Breakdown ({region}) - {scenario}{' -- ' + title if title else ''}")
        ax_bar.grid(axis="y", alpha=0.3)

        # Pie chart -- percentage breakdown
        nonzero = [
            (label, size_mb, size_bytes, color)
            for label, size_mb, size_bytes, color in zip(labels, sizes_mb, sizes_bytes, colors)
            if size_bytes > 0
        ]
        if nonzero:
            pie_labels, pie_sizes, _, pie_colors = zip(*nonzero)
            pie_labels = tuple(
                f"{label}\n{self._fmt_bytes(size_bytes)}"
                for label, _, size_bytes, _ in nonzero
            )
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
