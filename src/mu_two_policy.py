from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_prof import NodeType


@dataclass
class PolicyConfig:
    memory_limit_bytes: Optional[int] = None
    max_simulation_candidates: int = 64
    max_selection_steps: int = 16
    min_candidate_mem_bytes: int = 1 * 1024 * 1024
    knee_utility_ratio: float = 0.2


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


@dataclass
class PlanValidationReport:
    """Human-readable diagnostics for a policy-generated checkpoint plan."""

    checkpointable_activations: int
    eligible_candidates: int
    retained_nodes: int
    recompute_nodes: int
    min_candidate_mem_bytes: int
    estimated_peak_delta_bytes: int
    largest_candidates: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checkpointable_activations": self.checkpointable_activations,
            "eligible_candidates": self.eligible_candidates,
            "retained_nodes": self.retained_nodes,
            "recompute_nodes": self.recompute_nodes,
            "min_candidate_mem_bytes": self.min_candidate_mem_bytes,
            "estimated_peak_delta_bytes": self.estimated_peak_delta_bytes,
            "largest_candidates": self.largest_candidates,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def format_summary(self) -> str:
        lines = [
            "Checkpoint plan validation:",
            f"  ok={self.ok}",
            f"  checkpointable_activations={self.checkpointable_activations}",
            f"  eligible_candidates={self.eligible_candidates} "
            f"(min={_fmt_bytes(self.min_candidate_mem_bytes)})",
            f"  retained_nodes={self.retained_nodes}",
            f"  recompute_nodes={self.recompute_nodes}",
            f"  estimated_peak_delta={_fmt_bytes(max(0, self.estimated_peak_delta_bytes))}",
        ]
        if self.largest_candidates:
            lines.append("  largest_candidates:")
            for row in self.largest_candidates[:5]:
                lines.append(f"    - {row['name']}: {_fmt_bytes(row['memory_bytes'])}")
        if self.warnings:
            lines.append("  warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        if self.errors:
            lines.append("  errors:")
            for error in self.errors:
                lines.append(f"    - {error}")
        return "\n".join(lines)


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


def _simulate_peak_with_recompute_set(
    graph_profiler: Any,
    all_activations: Set[Any],
    recompute_set: Set[Any],
    first_bw_use: Dict[Any, Any],
    decomposed: Set[Any],
) -> int:
    """Return simulated peak memory for a proposed recompute set.

    This uses GraphProfiler's checkpoint-aware lifetime model so selection is
    driven by true peak reduction rather than summed activation sizes.
    """
    checkpoint_plan = SimpleNamespace(
        retained_nodes=set(all_activations) - set(recompute_set),
        recompute_nodes=set(recompute_set),
        first_backward_use=dict(first_bw_use),
    )

    alive_with_cp = graph_profiler._build_alive_ranges_with_checkpoint(
        decomposed,
        checkpoint_plan,
    )
    peak_after, _ = graph_profiler.compute_peak_memory_filtered(alive_with_cp)
    return int(peak_after)


def _select_recompute_nodes(
    graph_profiler: Any,
    candidates: List[CandidateRow],
    retained: Set[Any],
    recompute: Set[Any],
    first_bw_use: Dict[Any, Any],
    required_inputs_map: Dict[Any, Set[Any]],
    config: PolicyConfig,
) -> Tuple[int, float, int]:
    """Choose recompute set via iterative marginal peak-reduction utility.

    At each iteration, evaluate each remaining candidate by simulating the
    checkpoint-aware peak and selecting the candidate with best
    ``delta_peak / recompute_time_ms``.
    """
    estimated_peak_before, _ = graph_profiler.compute_peak_memory()
    if not candidates:
        return 0, 0.0, estimated_peak_before

    usable = [row for row in candidates if row.mem >= max(0, config.min_candidate_mem_bytes)]
    if not usable:
        return 0, 0.0, estimated_peak_before

    ordered = sorted(
        usable,
        key=lambda row: (
            row.mem / max(row.recompute_time_ms, 1e-6),
            row.mem,
            -graph_profiler.node_index.get(row.node, 0),
        ),
        reverse=True,
    )
    if config.max_simulation_candidates > 0:
        ordered = ordered[: config.max_simulation_candidates]

    all_activations = set(retained)
    remaining: List[CandidateRow] = list(ordered)
    decomposed = graph_profiler._find_decomposed_parents()

    current_peak = int(estimated_peak_before)
    selected_recompute_ms = 0.0
    selected_peak_drop = 0
    accepted_utilities: List[float] = []
    selected_steps = 0

    while remaining and (config.max_selection_steps <= 0 or selected_steps < config.max_selection_steps):
        best_row: Optional[CandidateRow] = None
        best_trial_peak = current_peak
        best_trial_first_bw: Dict[Any, Any] = {}
        best_delta_peak = 0
        best_utility = float("-inf")

        for row in remaining:
            trial_recompute = set(recompute)
            trial_recompute.add(row.node)

            trial_first_bw = dict(first_bw_use)
            if row.fbw_node is not None:
                trial_first_bw[row.node] = row.fbw_node

            trial_peak = _simulate_peak_with_recompute_set(
                graph_profiler=graph_profiler,
                all_activations=all_activations,
                recompute_set=trial_recompute,
                first_bw_use=trial_first_bw,
                decomposed=decomposed,
            )

            delta_peak = current_peak - trial_peak
            if delta_peak <= 0:
                continue

            utility = float(delta_peak) / max(row.recompute_time_ms, 1e-6)
            if (
                utility > best_utility
                or (
                    utility == best_utility
                    and (
                        delta_peak > best_delta_peak
                        or (
                            delta_peak == best_delta_peak
                            and row.mem > (best_row.mem if best_row is not None else -1)
                        )
                    )
                )
            ):
                best_row = row
                best_trial_peak = trial_peak
                best_trial_first_bw = trial_first_bw
                best_delta_peak = delta_peak
                best_utility = utility

        if best_row is None:
            break

        # Automatic diminishing-returns stop (knee-like behavior).
        if accepted_utilities:
            baseline_utility = accepted_utilities[0]
            if baseline_utility > 0.0 and best_utility < config.knee_utility_ratio * baseline_utility:
                break

        recompute.add(best_row.node)
        retained.discard(best_row.node)
        selected_recompute_ms += best_row.recompute_time_ms

        if best_row.fbw_node is not None:
            first_bw_use.clear()
            first_bw_use.update(best_trial_first_bw)
        required_inputs_map[best_row.node] = set(best_row.req_inputs)

        current_peak = best_trial_peak
        selected_peak_drop = int(estimated_peak_before) - int(current_peak)
        accepted_utilities.append(best_utility)
        remaining = [row for row in remaining if row.node is not best_row.node]
        selected_steps += 1

    return max(0, selected_peak_drop), selected_recompute_ms, int(current_peak)


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


def validate_checkpoint_plan(
    graph_profiler: Any,
    plan: CheckpointPlan,
    config: Optional[PolicyConfig] = None,
) -> PlanValidationReport:
    """Validate whether a checkpoint plan is implementable by the graph rewriter.

    This is intentionally diagnostic rather than fatal: the benchmark prints the
    report so we can distinguish "no useful checkpoint candidates" from "the
    planner emitted a malformed plan".
    """
    config = config or PolicyConfig()
    _validate_profiler_contract(graph_profiler)

    graph_nodes = set(graph_profiler.node_list)
    activations = [
        n
        for n in graph_profiler.intermediate_nodes
        if graph_profiler.first_bw_access.get(n) is not None
    ]
    min_candidate_mem = max(0, int(config.min_candidate_mem_bytes))
    eligible = [
        n
        for n in activations
        if int(graph_profiler.node_mem_bytes.get(n.name, 0)) >= min_candidate_mem
    ]
    largest_candidates = [
        {
            "name": n.name,
            "memory_bytes": int(graph_profiler.node_mem_bytes.get(n.name, 0)),
            "first_backward_use": (
                graph_profiler.first_bw_access[n].name
                if graph_profiler.first_bw_access.get(n) is not None
                else None
            ),
        }
        for n in sorted(
            activations,
            key=lambda node: int(graph_profiler.node_mem_bytes.get(node.name, 0)),
            reverse=True,
        )[:10]
    ]

    errors: List[str] = []
    warnings: List[str] = []
    retained = set(plan.retained_nodes)
    recompute = set(plan.recompute_nodes)

    overlap = retained & recompute
    if overlap:
        overlap_names = sorted(n.name for n in overlap if hasattr(n, "name"))
        errors.append(
            "Nodes appear in both retained_nodes and recompute_nodes: "
            + ", ".join(overlap_names[:10])
        )

    for node in sorted(recompute, key=lambda n: graph_profiler.node_index.get(n, 10**9)):
        if node not in graph_nodes:
            errors.append(f"Recompute target {node.name} is not in the graph")
            continue

        if graph_profiler.node_type.get(node) != NodeType.ACT:
            errors.append(f"Recompute target {node.name} is not classified as ACT")

        mem = int(graph_profiler.node_mem_bytes.get(node.name, 0))
        if mem <= 0:
            warnings.append(f"Recompute target {node.name} has zero measured output memory")

        fbw = plan.first_backward_use.get(node)
        if fbw is None:
            errors.append(f"Recompute target {node.name} has no first_backward_use")
        elif fbw not in graph_nodes:
            errors.append(f"first_backward_use for {node.name} is not in the graph")
        elif graph_profiler.node_index[fbw] < graph_profiler.sep_bw_idx:
            errors.append(
                f"first_backward_use for {node.name} occurs before the backward region"
            )

        required_inputs = plan.required_recompute_inputs.get(node)
        if not required_inputs:
            errors.append(f"Recompute target {node.name} has no required_recompute_inputs")
            continue

        invalid_inputs: List[str] = []
        for inp in required_inputs:
            inp_type = graph_profiler.node_type.get(inp, NodeType.OTHER)
            allowed = (
                inp in graph_nodes
                and (
                    inp.op == "placeholder"
                    or inp_type == NodeType.PARAM
                    or inp in retained
                )
            )
            if not allowed:
                invalid_inputs.append(
                    f"{getattr(inp, 'name', repr(inp))}:{getattr(inp_type, 'name', inp_type)}"
                )
        if invalid_inputs:
            errors.append(
                f"Recompute target {node.name} has invalid boundary inputs: "
                + ", ".join(invalid_inputs[:10])
            )

    peak_before = int(plan.estimated_peak_before_bytes or 0)
    peak_after = int(plan.estimated_peak_after_bytes or 0)
    estimated_peak_delta = max(0, peak_before - peak_after)

    if not recompute:
        if eligible:
            warnings.append(
                "Policy selected no recompute nodes even though eligible activation candidates exist"
            )
        else:
            warnings.append(
                "Policy selected no recompute nodes because no activation candidates met the size threshold"
            )
    elif estimated_peak_delta <= 0:
        warnings.append(
            "Policy selected recompute nodes but estimated peak memory did not decrease"
        )

    if int(plan.estimated_memory_saved_bytes) != estimated_peak_delta:
        warnings.append(
            "Plan saved-bytes summary does not match peak_before - peak_after"
        )

    return PlanValidationReport(
        checkpointable_activations=len(activations),
        eligible_candidates=len(eligible),
        retained_nodes=len(retained),
        recompute_nodes=len(recompute),
        min_candidate_mem_bytes=min_candidate_mem,
        estimated_peak_delta_bytes=estimated_peak_delta,
        largest_candidates=largest_candidates,
        errors=errors,
        warnings=warnings,
    )


def build_checkpoint_plan(
    graph_profiler: Any,
    config: Optional[PolicyConfig] = None,
) -> CheckpointPlan:
    """Build a checkpointing plan containing retained and recomputed activations."""

    config = config or PolicyConfig()
    _validate_profiler_contract(graph_profiler)

    activations = [
        n
        for n in graph_profiler.intermediate_nodes
        if graph_profiler.first_bw_access.get(n) is not None
    ]
    retained: Set[Any] = set(activations)
    recompute: Set[Any] = set()
    first_bw_use: Dict[Any, Any] = {}
    required_inputs_map: Dict[Any, Set[Any]] = {}

    estimated_peak_before, _ = graph_profiler.compute_peak_memory()
    candidates = _collect_policy_candidates(graph_profiler, activations)
    selected_mem_bytes, selected_recompute_ms, estimated_peak_after = _select_recompute_nodes(
        graph_profiler=graph_profiler,
        candidates=candidates,
        retained=retained,
        recompute=recompute,
        first_bw_use=first_bw_use,
        required_inputs_map=required_inputs_map,
        config=config,
    )

    # Report the implied memory target after algorithmic selection.
    memory_limit_bytes = max(0, estimated_peak_after)
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
