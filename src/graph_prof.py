import operator
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Tuple

import torch
import torch.fx as fx


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
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
        super().__init__(module, garbage_collect_values)


        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation.

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass.

        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.

        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The optimizer's states and the input mini-batch are also
        # placeholder nodes that given as inputs to the graph.

        # The parameters and gradients of the model can be otained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =



        self.device = torch.device("cuda")
        self.device_type = "cuda"

        # ===== STATIC ANALYSIS =====

        # Ordered node list and fast index lookup
        self.node_list: List[fx.Node] = list(self.module.graph.nodes)
        self.node_index: Dict[fx.Node, int] = {
            n: i for i, n in enumerate(self.node_list)
        }

        # 1. Locate sep / sep_backward boundary nodes
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

        # 2. Assign a region label to every node
        #    forward: placeholders + forward computation up to and including sep
        #    loss:    nodes between sep and sep_backward
        #    backward: sep_backward onwards (backward + optimizer + output)
        self.node_region: Dict[fx.Node, str] = {}
        for i, node in enumerate(self.node_list):
            if i <= self.sep_idx:
                self.node_region[node] = "forward"
            elif i < self.sep_bw_idx:
                self.node_region[node] = "loss"
            else:
                self.node_region[node] = "backward"

        # 3. Identify parameter, gradient, and optimizer-state nodes
        self.param_nodes: Set[fx.Node] = set()
        self.grad_nodes: Set[fx.Node] = set()
        self.optimizer_state_nodes: Set[fx.Node] = set()

        # Try _fused_adam first (CUDA path), then foreach heuristic (non-CUDA)
        if not self._identify_via_fused_adam():
            self._identify_via_foreach()

        # 4. Classify every node and identify intermediate activations
        self.node_type: Dict[fx.Node, NodeType] = {}
        self.intermediate_nodes: List[fx.Node] = []
        self.last_fw_access: Dict[fx.Node, Optional[fx.Node]] = {}
        self.first_bw_access: Dict[fx.Node, Optional[fx.Node]] = {}

        for node in self.node_list:
            if node in self.param_nodes:
                self.node_type[node] = NodeType.PARAM
                continue
            if node in self.grad_nodes:
                self.node_type[node] = NodeType.GRAD
                continue

            # Activation = forward non-placeholder node with at least one
            # backward user (used after sep_backward)
            if (
                self.node_region[node] == "forward"
                and node.op not in (OP.PLACEHOLDER, OP.OUTPUT)
            ):
                bw_users = [
                    u for u in node.users
                    if self.node_index[u] >= self.sep_bw_idx
                ]
                if bw_users:
                    self.intermediate_nodes.append(node)
                    self.node_type[node] = NodeType.ACT

                    # Last use in forward (+loss) region
                    fw_users = [
                        u for u in node.users
                        if self.node_index[u] < self.sep_bw_idx
                    ]
                    self.last_fw_access[node] = (
                        max(fw_users, key=lambda u: self.node_index[u])
                        if fw_users else None
                    )
                    # First use in backward region
                    self.first_bw_access[node] = min(
                        bw_users, key=lambda u: self.node_index[u]
                    )
                    continue

            self.node_type[node] = NodeType.OTHER

        # ===== RUNTIME PROFILING STORAGE =====
        self.node_runtimes: Dict[str, List[float]] = {
            n.name: [] for n in self.node_list
        }
        self.node_mem_bytes: Dict[str, int] = {}
        self.node_avg_runtime: Dict[str, float] = {}

    # ----- param / grad identification strategies -----

    def _identify_via_fused_adam(self) -> bool:
        """CUDA path: extract params/grads/opt-states from _fused_adam node."""
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
        """Non-CUDA path: infer params/opt-states/grads from graph patterns.

        Parameters are placeholders that:
          (a) are written to by copy_ in the optimizer section, AND
          (b) have users in the forward pass (they participate in forward compute).

        Optimizer states are placeholders that:
          (a) are written to by copy_, but
          (b) have NO forward-pass users.

        Gradients are identified from _foreach_addcmul.Scalar which computes
        exp_avg_sq += (1-beta2) * grad * grad  — args[1] is the gradient list.
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

    # ----- runtime hooks -----

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: fx.Node) -> Any:
        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        # you can start measuring the run-time of a node here


        # ----- timing start -----
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        result = super().run_node(n)
          # you can end measuring the run-time of a node here HINT:
        # For CUDA: use torch.cuda.Event(enable_timing=True) for GPU timing.

        # ----- timing end -----
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_ms = start_evt.elapsed_time(end_evt)

        self.node_runtimes[n.name].append(elapsed_ms)

        # ----- memory (measure once on first profiling iteration) -----
        if n.name not in self.node_mem_bytes:
            self.node_mem_bytes[n.name] = self._tensor_memory(result)

        return result

    # ----- stats lifecycle -----

    def reset_stats(self) -> None:
        for name in self.node_runtimes:
            self.node_runtimes[name] = []
        self.node_avg_runtime = {}

    def aggregate_stats(self) -> None:
        for name, times in self.node_runtimes.items():
            self.node_avg_runtime[name] = (
                sum(times) / len(times) if times else 0.0
            )

    def print_stats(self) -> None:
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

    # ----- peak memory analysis -----

    def compute_peak_memory(self) -> Tuple[int, Dict[NodeType, int]]:
        """
        Analytically compute peak memory by simulating tensor lifetimes.
        A tensor is alive from its production until its last consumer finishes.
        Returns (peak_total_bytes, {NodeType: bytes_at_peak_step}).
        """
        # Identify "decomposable" parents: nodes whose ALL users are getitem.
        # For these, we skip the parent and instead count each getitem child
        # individually so that each component gets its correct NodeType
        # (e.g. GRAD for gradient components vs OTHER for the parent tuple).
        decomposed_parents: Set[fx.Node] = set()
        for node in self.node_list:
            if (
                node.users
                and all(u.target is operator.getitem for u in node.users)
            ):
                decomposed_parents.add(node)

        # Build alive intervals: node -> (born_idx, dies_idx)
        # Parameters and gradients are treated as persistent — once allocated
        # they reside in memory throughout the iteration (per project spec).
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
            # Params and grads persist for the entire iteration:
            # param buffers + grad buffers are allocated once and reside in
            # memory throughout (see project spec). Set born=0, dies=last.
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

    # ----- helpers -----

    def plot_peak_memory_breakdown(self, title: str = "", save_path: Optional[str] = None) -> None:
        """Generate a bar chart of peak memory breakdown by NodeType."""
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

    @staticmethod
    def _tensor_memory(val: Any) -> int:
        """Return total bytes for a tensor or collection of tensors."""
        if isinstance(val, torch.Tensor):
            return val.nelement() * val.element_size()
        if isinstance(val, (tuple, list)):
            return sum(GraphProfiler._tensor_memory(v) for v in val)
        return 0

    @staticmethod
    def _fmt_bytes(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        if b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        if b < 1024 ** 3:
            return f"{b / 1024 ** 2:.1f} MB"
        return f"{b / 1024 ** 3:.2f} GB"
