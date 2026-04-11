from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_prof import NodeType


@dataclass
class PolicyConfig:
    memory_limit_bytes: Optional[int] = None


@dataclass
class CheckpointPlan:
    """Structured output consumed by graph rewriting and reporting."""

    retained_nodes: Set[Any]
    recompute_nodes: Set[Any]
    first_backward_use: Dict[Any, Any]
    required_recompute_inputs: Dict[Any, Set[Any]]
    estimated_memory_saved_bytes: int = 0
    estimated_recompute_overhead_ms: float = 0.0
    estimated_peak_before_bytes: Optional[int] = None
    estimated_peak_after_bytes: Optional[int] = None
    memory_limit_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the plan without raw FX node objects for later analysis."""
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
            "memory_limit_bytes": self.memory_limit_bytes,
        }


@dataclass
class CandidateRow:
    node: Any
    mem: int
    recompute_time_ms: float
    fbw_node: Optional[Any]
    req_inputs: Set[Any]


def _collect_policy_candidates(graph_profiler: Any, activations: List[Any]) -> List[CandidateRow]:
    """Build per-activation candidate rows used by the scheduler."""

    candidates: List[CandidateRow] = []
    for node in activations:
        mem = int(graph_profiler.node_mem_bytes.get(node.name, 0))
        fbw = graph_profiler.first_bw_access.get(node)
        required_inputs, recompute_time_ms = _estimate_recompute_metrics(node, graph_profiler)

        candidates.append(
            CandidateRow(
                node=node,
                mem=mem,
                recompute_time_ms=recompute_time_ms,
                fbw_node=fbw,
                req_inputs=required_inputs,
            )
        )
    return candidates


def _select_recompute_nodes(
    graph_profiler: Any,
    candidates: List[CandidateRow],
    memory_limit_bytes: int,
    estimated_peak_before: int,
    retained: Set[Any],
    recompute: Set[Any],
    first_bw_use: Dict[Any, Any],
    required_inputs_map: Dict[Any, Set[Any]],
) -> Tuple[int, float]:
    selected_mem_bytes = 0
    selected_recompute_ms = 0.0

    ordered = sorted(
        candidates,
        key=lambda row: (
            row.mem / max(row.recompute_time_ms, 1e-6),
            row.mem,
            -graph_profiler.node_index.get(row.node, 0),
        ),
        reverse=True,
    )

    for row in ordered:
        current_peak_after_selection = max(0, estimated_peak_before - selected_mem_bytes)
        selected = current_peak_after_selection > memory_limit_bytes

        if selected:
            selected_mem_bytes += row.mem
            recompute.add(row.node)
            retained.discard(row.node)
            selected_recompute_ms += row.recompute_time_ms
            if row.fbw_node is not None:
                first_bw_use[row.node] = row.fbw_node
                required_inputs_map[row.node] = set(row.req_inputs)
    return selected_mem_bytes, selected_recompute_ms


def _estimate_recompute_metrics(
    target_node: Any,
    graph_profiler: Any,
) -> Tuple[Set[Any], float]:
    """Estimate the recompute boundary inputs and total runtime cost for one activation."""

    required_inputs: Set[Any] = set()
    recompute_nodes: Set[Any] = set()
    visited: Set[Any] = set()

    def visit(node: Any) -> None:
        if node in visited:
            return
        visited.add(node)

        region = graph_profiler.node_region.get(node)
        if region != "forward":
            return

        if node.op == "placeholder":
            required_inputs.add(node)
            return

        node_type = graph_profiler.node_type.get(node)
        if node_type == NodeType.PARAM:
            required_inputs.add(node)
            return

        recompute_nodes.add(node)
        for inp in node.all_input_nodes:
            visit(inp)

    visit(target_node)

    recompute_time_ms = sum(
        float(graph_profiler.node_avg_runtime.get(node.name, 0.0))
        for node in recompute_nodes
    )
    return required_inputs, recompute_time_ms


def _validate_profiler_contract(graph_profiler: Any) -> None:
    required_attrs = [
        "node_list",
        "node_index",
        "node_type",
        "node_region",
        "intermediate_nodes",
        "first_bw_access",
        "last_fw_access",
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
    """Build a checkpointing plan containing retained and recomputed activations."""

    config = config or PolicyConfig()
    _validate_profiler_contract(graph_profiler)

    activations = list(graph_profiler.intermediate_nodes)
    retained: Set[Any] = set(activations)
    recompute: Set[Any] = set()
    first_bw_use: Dict[Any, Any] = {}
    required_inputs_map: Dict[Any, Set[Any]] = {}

    estimated_peak_before, _ = graph_profiler.compute_peak_memory()
    memory_limit_bytes = config.memory_limit_bytes
    if memory_limit_bytes is None:
        memory_limit_bytes = int(estimated_peak_before * 0.75)

    candidates = _collect_policy_candidates(graph_profiler, activations)
    selected_mem_bytes, selected_recompute_ms = _select_recompute_nodes(
        graph_profiler=graph_profiler,
        candidates=candidates,
        memory_limit_bytes=memory_limit_bytes,
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
        estimated_memory_saved_bytes=selected_mem_bytes,
        estimated_recompute_overhead_ms=selected_recompute_ms,
        estimated_peak_before_bytes=estimated_peak_before,
        estimated_peak_after_bytes=estimated_peak_after,
        memory_limit_bytes=memory_limit_bytes,
    )