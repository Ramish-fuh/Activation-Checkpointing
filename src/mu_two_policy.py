from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_prof import NodeType


@dataclass
class PolicyConfig:
    max_recompute_overhead_ms: Optional[float] = None
    min_memory_saved_bytes: int = 0
    # Global peak-memory target: stop selecting more nodes once the estimate
    # is low enough, trading extra recompute for less memory.
    target_peak_memory_bytes: Optional[int] = None


@dataclass
class NodeDecision:
    """Per-node decision trace for debugging and reporting."""
    node_name: str
    selected_for_recompute: bool
    memory_saved_bytes: int
    recompute_cost_ms: float
    score: float
    first_backward_use_name: Optional[str]
    required_input_names: List[str]
    skip_reason: Optional[str] = None


@dataclass
class CheckpointPlan:
    """Structured output consumed by graph rewriting and reporting."""
    retained_nodes: Set[Any]
    recompute_nodes: Set[Any]
    first_backward_use: Dict[Any, Any]
    required_recompute_inputs: Dict[Any, Set[Any]]
    decisions: List[NodeDecision] = field(default_factory=list)
    estimated_memory_saved_bytes: int = 0
    estimated_recompute_overhead_ms: float = 0.0
    estimated_peak_before_bytes: Optional[int] = None
    estimated_peak_after_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the plan without raw FX node objects."""
        return {
            "retained_nodes": sorted(n.name for n in self.retained_nodes),
            "recompute_nodes": sorted(n.name for n in self.recompute_nodes),
            "first_backward_use": {
                n.name: bw.name for n, bw in self.first_backward_use.items()
            },
            "required_recompute_inputs": {
                n.name: sorted(inp.name for inp in inputs)
                for n, inputs in self.required_recompute_inputs.items()
            },
            "estimated_memory_saved_bytes": self.estimated_memory_saved_bytes,
            "estimated_recompute_overhead_ms": self.estimated_recompute_overhead_ms,
            "estimated_peak_before_bytes": self.estimated_peak_before_bytes,
            "estimated_peak_after_bytes": self.estimated_peak_after_bytes,
            "decisions": [
                {
                    "node_name": d.node_name,
                    "selected_for_recompute": d.selected_for_recompute,
                    "memory_saved_bytes": d.memory_saved_bytes,
                    "recompute_cost_ms": d.recompute_cost_ms,
                    "score": d.score,
                    "first_backward_use_name": d.first_backward_use_name,
                    "required_input_names": list(d.required_input_names),
                    "skip_reason": d.skip_reason,
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
                reason = f" ({d.skip_reason})" if d.skip_reason else ""
                print(
                    f"    {d.node_name:30s} {status:10s} "
                    f"score={d.score:10.4f} mem={_fmt_bytes(d.memory_saved_bytes):>10s} "
                    f"cost={d.recompute_cost_ms:8.4f}ms{reason}"
                )


CandidateRow = Tuple[Any, int, float, float, Optional[str], Set[Any], Optional[Any]]


def _collect_policy_candidates(graph_profiler: Any, activations: List[Any]) -> List[CandidateRow]:
    candidates: List[CandidateRow] = []
    for node in activations:
        mem = int(graph_profiler.node_mem_bytes.get(node.name, 0))
        cost = float(graph_profiler.node_avg_runtime.get(node.name, 0.0))
        fbw = graph_profiler.first_bw_access.get(node)
        required_inputs, legality_issue = _required_recompute_inputs(node, graph_profiler)

        if fbw is None:
            candidates.append(
                (node, mem, cost, 0.0, "no_backward_use", required_inputs, None)
            )
            continue

        if legality_issue is not None:
            candidates.append(
                (node, mem, cost, 0.0, legality_issue, required_inputs, fbw)
            )
            continue

        score = float(mem) / max(cost, 1e-6)
        candidates.append((node, mem, cost, score, None, required_inputs, fbw))
    return candidates


def _policy_skip_reason(
    config: PolicyConfig,
    pre_skip_reason: Optional[str],
    target_peak_reached: bool,
    mem: int,
    cost: float,
    selected_cost_ms: float,
    req_inputs: Set[Any],
    recompute: Set[Any],
) -> Optional[str]:
    if pre_skip_reason is not None:
        return pre_skip_reason
    if target_peak_reached:
        return "target_peak_reached"
    if mem < config.min_memory_saved_bytes:
        return "below_min_memory_threshold"
    if (
        config.max_recompute_overhead_ms is not None
        and selected_cost_ms + cost > config.max_recompute_overhead_ms
    ):
        return "over_recompute_budget"
    if req_inputs.intersection(recompute):
        return "depends_on_recomputed_activation"
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

    for node, mem, cost, score, pre_skip_reason, req_inputs, fbw_node in candidates:
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
) -> Tuple[Set[Any], Optional[str]]:
    """Return legal boundary inputs needed to recompute an activation.

    Boundary inputs are placeholders and retained activations (other than
    the target node itself). Any dependency crossing into non-forward region
    is treated as illegal for conservative rewriting.
    """
    required_inputs: Set[Any] = set()
    visited: Set[Any] = set()

    def visit(node: Any) -> Optional[str]:
        if node in visited:
            return None
        visited.add(node)

        region = graph_profiler.node_region.get(node)
        if region != "forward":
            return "non_forward_dependency"

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
            row[3],
            row[1],
            -graph_profiler.node_index[row[0]],
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
