from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_prof import NodeType


@dataclass
class PolicyConfig:
    max_recompute_overhead_ms: Optional[float] = None
    min_memory_saved_bytes: int = 0
    # Global peak-memory target: stop selecting more nodes once the estimate
    # is low enough, trading extra recompute for less memory.
    target_peak_memory_bytes: Optional[int] = None


class SkipReason(str, Enum):
    NO_BACKWARD_USE = "no_backward_use"
    NON_FORWARD_DEPENDENCY = "non_forward_dependency" #  would need something outside the allowed forward region.
    TARGET_PEAK_REACHED = "target_peak_reached"
    BELOW_MIN_MEMORY_THRESHOLD = "below_min_memory_threshold" # would save too little memory to be worth it.
    OVER_RECOMPUTE_BUDGET = "over_recompute_budget" # would exceed user-configured recompute overhead limit.
    DEPENDS_ON_RECOMPUTED_ACTIVATION = "depends_on_recomputed_activation"


class PlanDictKey(str, Enum):
    RETAINED_NODES = "retained_nodes"
    RECOMPUTE_NODES = "recompute_nodes"
    FIRST_BACKWARD_USE = "first_backward_use"
    REQUIRED_RECOMPUTE_INPUTS = "required_recompute_inputs"
    ESTIMATED_MEMORY_SAVED_BYTES = "estimated_memory_saved_bytes"
    ESTIMATED_RECOMPUTE_OVERHEAD_MS = "estimated_recompute_overhead_ms"
    ESTIMATED_PEAK_BEFORE_BYTES = "estimated_peak_before_bytes"
    ESTIMATED_PEAK_AFTER_BYTES = "estimated_peak_after_bytes"
    DECISIONS = "decisions"


class DecisionDictKey(str, Enum):
    NODE_NAME = "node_name"
    SELECTED_FOR_RECOMPUTE = "selected_for_recompute"
    MEMORY_SAVED_BYTES = "memory_saved_bytes"
    RECOMPUTE_COST_MS = "recompute_cost_ms"
    SCORE = "score"
    FIRST_BACKWARD_USE_NAME = "first_backward_use_name"
    REQUIRED_INPUT_NAMES = "required_input_names"
    SKIP_REASON = "skip_reason"


@dataclass
class NodeDecision:
    """Per-node decision trace for debugging and reporting."""
    node_name: str
    selected_for_recompute: bool
    memory_saved_bytes: int
    recompute_cost_ms: float
    score: float # ratio of memory_saved_bytes to recompute_cost_ms
    first_backward_use_name: Optional[str]
    required_input_names: List[str] # names of boundary inputs required to recompute this node
    skip_reason: Optional[SkipReason] = None


@dataclass
class CheckpointPlan:
    """Structured output consumed by graph rewriting and reporting."""
    retained_nodes: Set[Any]
    recompute_nodes: Set[Any]
    first_backward_use: Dict[Any, Any]  # recompute node -> first backward consumer
    required_recompute_inputs: Dict[Any, Set[Any]]  # recompute node -> boundary inputs
    decisions: List[NodeDecision] = field(default_factory=list)  # per-node decision trace
    estimated_memory_saved_bytes: int = 0  # summed estimated memory saved
    estimated_recompute_overhead_ms: float = 0.0  # summed recompute runtime overhead
    estimated_peak_before_bytes: Optional[int] = None  # baseline simulated peak
    estimated_peak_after_bytes: Optional[int] = None  # peak estimate after selected recompute

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the plan without raw FX node objects for later analysis."""
        return {
            PlanDictKey.RETAINED_NODES.value: sorted(n.name for n in self.retained_nodes),
            PlanDictKey.RECOMPUTE_NODES.value: sorted(n.name for n in self.recompute_nodes),
            PlanDictKey.FIRST_BACKWARD_USE.value: {
                n.name: bw.name for n, bw in self.first_backward_use.items()
            },
            PlanDictKey.REQUIRED_RECOMPUTE_INPUTS.value: {
                n.name: sorted(inp.name for inp in inputs)
                for n, inputs in self.required_recompute_inputs.items()
            },
            PlanDictKey.ESTIMATED_MEMORY_SAVED_BYTES.value: self.estimated_memory_saved_bytes,
            PlanDictKey.ESTIMATED_RECOMPUTE_OVERHEAD_MS.value: self.estimated_recompute_overhead_ms,
            PlanDictKey.ESTIMATED_PEAK_BEFORE_BYTES.value: self.estimated_peak_before_bytes,
            PlanDictKey.ESTIMATED_PEAK_AFTER_BYTES.value: self.estimated_peak_after_bytes,
            PlanDictKey.DECISIONS.value: [
                {
                    DecisionDictKey.NODE_NAME.value: d.node_name,
                    DecisionDictKey.SELECTED_FOR_RECOMPUTE.value: d.selected_for_recompute,
                    DecisionDictKey.MEMORY_SAVED_BYTES.value: d.memory_saved_bytes,
                    DecisionDictKey.RECOMPUTE_COST_MS.value: d.recompute_cost_ms,
                    DecisionDictKey.SCORE.value: d.score,
                    DecisionDictKey.FIRST_BACKWARD_USE_NAME.value: d.first_backward_use_name,
                    DecisionDictKey.REQUIRED_INPUT_NAMES.value: list(d.required_input_names),
                    DecisionDictKey.SKIP_REASON.value: d.skip_reason.value if d.skip_reason is not None else None,
                }
                for d in self.decisions
            ],
        }

    def print_summary(self, top_k: int = 10) -> None:
        print("\n--- Phase 2 Checkpoint Plan ---")
        print(f"  Recompute nodes: {len(self.recompute_nodes)}")
        print(f"  Retained nodes : {len(self.retained_nodes)}")
        print(
            f"  Estimated saved memory: {_fmt_bytes(self.estimated_memory_saved_bytes)}"
        )
        print(
            "  Estimated recompute overhead: "
            f"{self.estimated_recompute_overhead_ms:.4f} ms"
        )
        if self.estimated_peak_before_bytes is not None:
            print(
                "  Estimated peak (before -> after): "
                f"{_fmt_bytes(self.estimated_peak_before_bytes)} -> "
                f"{_fmt_bytes(self.estimated_peak_after_bytes or 0)}"
            )

        shown = self.decisions[:top_k]
        if shown:
            print("\n  Top decisions:")
            for d in shown:
                status = "RECOMPUTE" if d.selected_for_recompute else "retain"
                reason = f" ({d.skip_reason.value})" if d.skip_reason else ""
                print(
                    f"    {d.node_name:30s} {status:10s} "
                    f"score={d.score:10.4f} mem={_fmt_bytes(d.memory_saved_bytes):>10s} "
                    f"cost={d.recompute_cost_ms:8.4f}ms{reason}"
                )


@dataclass
class CandidateRow:
    node: Any
    mem: int
    cost: float
    score: float
    pre_skip_reason: Optional[SkipReason] #gets set during candidate collection in _collect_policy_candidates based on static legality checks. eg. no backward use, non-forward dependency, etc.
    req_inputs: Set[Any]
    fbw_node: Optional[Any]


def _collect_policy_candidates(graph_profiler: Any, activations: List[Any]) -> List[CandidateRow]:
    candidates: List[CandidateRow] = []
    for node in activations:
        mem = int(graph_profiler.node_mem_bytes.get(node.name, 0))
        cost = float(graph_profiler.node_avg_runtime.get(node.name, 0.0))
        fbw = graph_profiler.first_bw_access.get(node)
        required_inputs, pre_skip_reason = _required_recompute_inputs(node, graph_profiler)

        if fbw is None:
            candidates.append(
                CandidateRow(
                    node=node,
                    mem=mem,
                    cost=cost,
                    score=0.0,
                    pre_skip_reason=SkipReason.NO_BACKWARD_USE,
                    req_inputs=required_inputs,
                    fbw_node=None,
                )
            )
            continue

        if pre_skip_reason is not None:
            candidates.append(
                CandidateRow(
                    node=node,
                    mem=mem,
                    cost=cost,
                    score=0.0,
                    pre_skip_reason=pre_skip_reason,
                    req_inputs=required_inputs,
                    fbw_node=fbw,
                )
            )
            continue

        score = float(mem) / max(cost, 1e-6)
        candidates.append(
            CandidateRow(
                node=node,
                mem=mem,
                cost=cost,
                score=score,
                pre_skip_reason=None,
                req_inputs=required_inputs,
                fbw_node=fbw,
            )
        )
    return candidates


def _policy_skip_reason(
    config: PolicyConfig,
    pre_skip_reason: Optional[SkipReason],
    target_peak_reached: bool,
    mem: int,
    cost: float,
    selected_cost_ms: float,
    req_inputs: Set[Any],
    recompute: Set[Any],
) -> Optional[SkipReason]:
    if pre_skip_reason is not None:
        return pre_skip_reason
    if target_peak_reached:
        return SkipReason.TARGET_PEAK_REACHED
    if mem < config.min_memory_saved_bytes:
        return SkipReason.BELOW_MIN_MEMORY_THRESHOLD
    if (
        config.max_recompute_overhead_ms is not None
        and selected_cost_ms + cost > config.max_recompute_overhead_ms
    ):
        return SkipReason.OVER_RECOMPUTE_BUDGET
    if req_inputs.intersection(recompute):
        return SkipReason.DEPENDS_ON_RECOMPUTED_ACTIVATION
    return None


def _select_recompute_nodes(
    candidates: List[CandidateRow],
    config: PolicyConfig,
    estimated_peak_before: int,
    retained: Set[Any],
    recompute: Set[Any],
    first_bw_use: Dict[Any, Any],
    required_inputs_map: Dict[Any, Set[Any]],
) -> Tuple[List[NodeDecision], int, float]:
    decisions: List[NodeDecision] = []
    selected_mem_bytes = 0
    selected_cost_ms = 0.0
    target_peak_reached = False

    for row in candidates:
        node = row.node
        mem = row.mem
        cost = row.cost
        score = row.score
        pre_skip_reason = row.pre_skip_reason
        req_inputs = row.req_inputs
        fbw_node = row.fbw_node
        skip_reason = _policy_skip_reason(
            config=config,
            pre_skip_reason=pre_skip_reason,
            target_peak_reached=target_peak_reached,
            mem=mem,
            cost=cost,
            selected_cost_ms=selected_cost_ms,
            req_inputs=req_inputs,
            recompute=recompute,
        )
        selected = skip_reason is None

        if selected:
            recompute.add(node)
            retained.discard(node)
            selected_mem_bytes += mem
            selected_cost_ms += cost
            if fbw_node is not None:
                first_bw_use[node] = fbw_node
                required_inputs_map[node] = set(req_inputs)
            if config.target_peak_memory_bytes is not None:
                estimated_after = max(0, estimated_peak_before - selected_mem_bytes)
                if estimated_after <= config.target_peak_memory_bytes:
                    target_peak_reached = True

        decisions.append(
            NodeDecision(
                node_name=node.name,
                selected_for_recompute=selected,
                memory_saved_bytes=mem,
                recompute_cost_ms=cost,
                score=score,
                first_backward_use_name=fbw_node.name if fbw_node is not None else None,
                required_input_names=sorted(inp.name for inp in req_inputs),
                skip_reason=skip_reason,
            )
        )

    decisions.sort(key=lambda d: d.score, reverse=True)
    return decisions, selected_mem_bytes, selected_cost_ms


def _required_recompute_inputs(
    target_node: Any,
    graph_profiler: Any,
) -> Tuple[Set[Any], Optional[SkipReason]]:
    """Return legal boundary inputs needed to recompute an activation.

    Boundary inputs are placeholders and retained activations (other than
    the target node itself). Any dependency crossing into non-forward region
    is treated as illegal for conservative rewriting.
    """
    required_inputs: Set[Any] = set()
    visited: Set[Any] = set()

    def visit(node: Any) -> Optional[SkipReason]:
        if node in visited:
            return None
        visited.add(node)

        region = graph_profiler.node_region.get(node)
        if region != "forward":
            return SkipReason.NON_FORWARD_DEPENDENCY

        if node.op == "placeholder":
            required_inputs.add(node)
            return None

        node_type = graph_profiler.node_type.get(node)
        if node_type == NodeType.PARAM:
            # Parameters are placeholders in this graph style, but keep this
            # branch for safety if classification or tracing changes.
            required_inputs.add(node)
            return None

        if node_type == NodeType.ACT and node is not target_node:
            required_inputs.add(node)
            return None

        for inp in node.all_input_nodes:
            issue = visit(inp)
            if issue is not None:
                return issue
        return None

    for parent in target_node.all_input_nodes:
        issue = visit(parent)
        if issue is not None:
            return set(), issue

    return required_inputs, None


def _validate_profiler_contract(graph_profiler: Any) -> None:
    required_attrs = [
        "node_list",
        "node_index",
        "node_type",
        "node_region",
        "intermediate_nodes",
        "first_bw_access",
        "node_avg_runtime",
        "node_mem_bytes",
    ]
    missing = [name for name in required_attrs if not hasattr(graph_profiler, name)]
    if missing:
        raise ValueError(
            "GraphProfiler missing required attributes for policy: " + ", ".join(missing)
        )


def _fmt_bytes(b: int) -> str:
    for unit, threshold in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if b >= threshold:
            return f"{b / threshold:.1f} {unit}"
    return f"{b} B"


def build_checkpoint_plan(
    graph_profiler: Any,
    config: Optional[PolicyConfig] = None,
) -> CheckpointPlan:
    """Build a conservative, mu-TWO-style checkpointing plan.

    The policy prioritizes activations that save more memory per unit
    recomputation cost and enforces legality constraints for Phase 3.
    """
    config = config or PolicyConfig()
    _validate_profiler_contract(graph_profiler)

    activations = list(graph_profiler.intermediate_nodes)
    retained: Set[Any] = set(activations)
    recompute: Set[Any] = set()
    first_bw_use: Dict[Any, Any] = {}
    required_inputs_map: Dict[Any, Set[Any]] = {}

    estimated_peak_before, _ = graph_profiler.compute_peak_memory()
    candidates = _collect_policy_candidates(graph_profiler, activations)

    # Deterministic ordering: best score, then larger memory, then topological order.
    candidates.sort(
        key=lambda row: (
            row.score,
            row.mem,
            -graph_profiler.node_index[row.node],
        ),
        reverse=True,
    )
    decisions, selected_mem_bytes, selected_cost_ms = _select_recompute_nodes(
        candidates=candidates,
        config=config,
        estimated_peak_before=estimated_peak_before,
        retained=retained,
        recompute=recompute,
        first_bw_use=first_bw_use,
        required_inputs_map=required_inputs_map,
    )

    estimated_peak_after = max(0, estimated_peak_before - selected_mem_bytes)
    return CheckpointPlan(
        retained_nodes=retained,
        recompute_nodes=recompute,
        first_backward_use=first_bw_use,
        required_recompute_inputs=required_inputs_map,
        decisions=decisions,
        estimated_memory_saved_bytes=selected_mem_bytes,
        estimated_recompute_overhead_ms=selected_cost_ms,
        estimated_peak_before_bytes=estimated_peak_before,
        estimated_peak_after_bytes=estimated_peak_after,
    )
