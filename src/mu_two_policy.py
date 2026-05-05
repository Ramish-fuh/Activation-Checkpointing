from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_prof import NodeType


@dataclass
class PolicyConfig:
    memory_limit_bytes: Optional[int] = None
    default_memory_budget_fraction: float = 0.5
    optimize_region: str = "forward"
    reject_overall_peak_increase: bool = True
    max_simulation_candidates: int = 64
    max_selection_steps: int = 16
    min_candidate_mem_bytes: int = 1 * 1024 * 1024


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
    optimize_region: str = "forward"

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
            "optimize_region": self.optimize_region,
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
    excluded_param_only_candidates: int
    excluded_alias_candidates: int
    optimize_region: str
    estimated_peak_delta_bytes: int
    memory_limit_bytes: Optional[int]
    memory_budget_satisfied: Optional[bool]
    forward_peak_before_bytes: int
    forward_peak_after_bytes: int
    overall_peak_before_bytes: int
    overall_peak_after_bytes: int
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
            "excluded_param_only_candidates": self.excluded_param_only_candidates,
            "excluded_alias_candidates": self.excluded_alias_candidates,
            "optimize_region": self.optimize_region,
            "estimated_peak_delta_bytes": self.estimated_peak_delta_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_budget_satisfied": self.memory_budget_satisfied,
            "forward_peak_before_bytes": self.forward_peak_before_bytes,
            "forward_peak_after_bytes": self.forward_peak_after_bytes,
            "overall_peak_before_bytes": self.overall_peak_before_bytes,
            "overall_peak_after_bytes": self.overall_peak_after_bytes,
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
            f"  excluded_param_only_candidates={self.excluded_param_only_candidates}",
            f"  excluded_alias_candidates={self.excluded_alias_candidates}",
            f"  optimize_region={self.optimize_region}",
            f"  retained_nodes={self.retained_nodes}",
            f"  recompute_nodes={self.recompute_nodes}",
            f"  estimated_peak_delta={_fmt_bytes(max(0, self.estimated_peak_delta_bytes))}",
            f"  memory_limit="
            f"{_fmt_bytes(self.memory_limit_bytes) if self.memory_limit_bytes is not None else 'None'}",
            f"  memory_budget_satisfied={self.memory_budget_satisfied}",
            f"  forward_peak={_fmt_bytes(self.forward_peak_before_bytes)} -> "
            f"{_fmt_bytes(self.forward_peak_after_bytes)}",
            f"  overall_peak={_fmt_bytes(self.overall_peak_before_bytes)} -> "
            f"{_fmt_bytes(self.overall_peak_after_bytes)}",
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


def _is_param_only_candidate(graph_profiler: Any, row: CandidateRow) -> bool:
    """True when recomputing the target depends only on model parameters.

    Such targets are usually parameter views/transposes, not mini-batch
    activations. Checkpointing them would not represent activation memory
    savings and can produce misleading plans.
    """
    if not row.req_inputs:
        return False
    return all(graph_profiler.node_type.get(inp) == NodeType.PARAM for inp in row.req_inputs)


def _is_alias_candidate(graph_profiler: Any, row: CandidateRow) -> bool:
    """True when the target is a view-like metadata node.

    Activation checkpointing should target tensors that own activation storage.
    View/transpose/expand/slice nodes can be saved by autograd, but they mostly
    describe another tensor's storage and make poor checkpoint targets.
    """
    is_alias_node = getattr(graph_profiler, "_is_alias_node", None)
    if callable(is_alias_node) and is_alias_node(row.node):
        return True

    name = getattr(row.node, "name", "")
    alias_name_prefixes = (
        "view",
        "transpose",
        "expand",
        "slice",
        "select",
        "squeeze",
        "unsqueeze",
        "permute",
    )
    if name == "t" or name.startswith("t_"):
        return True
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in alias_name_prefixes)


def _compute_peak_from_alive(
    graph_profiler: Any,
    alive: Dict[Any, Tuple[int, int]],
    optimize_region: str,
) -> int:
    """Compute peak bytes in the policy's selected optimization region."""

    if optimize_region == "overall":
        peak, _ = graph_profiler.compute_peak_memory_filtered(alive)
        return int(peak)

    if optimize_region == "forward":
        _, peak, _ = graph_profiler._sweep_for_forward_peak(alive)
        return int(peak)

    raise ValueError(
        f"Unsupported optimize_region={optimize_region!r}; expected 'forward' or 'overall'"
    )


def _resolve_memory_limit_bytes(baseline_peak_bytes: int, config: PolicyConfig) -> int:
    """Return the target peak used by the budget-driven policy loop."""
    if config.memory_limit_bytes is not None:
        return max(0, int(config.memory_limit_bytes))

    fraction = max(0.0, min(1.0, float(config.default_memory_budget_fraction)))
    return int(baseline_peak_bytes * fraction)


def _build_alive_with_recompute_set(
    graph_profiler: Any,
    all_activations: Set[Any],
    recompute_set: Set[Any],
    first_bw_use: Dict[Any, Any],
    decomposed: Set[Any],
) -> Dict[Any, Tuple[int, int]]:
    """Build checkpoint-aware alive ranges for a proposed recompute set."""
    checkpoint_plan = SimpleNamespace(
        retained_nodes=set(all_activations) - set(recompute_set),
        recompute_nodes=set(recompute_set),
        first_backward_use=dict(first_bw_use),
    )

    alive_with_cp = graph_profiler._build_alive_ranges_with_checkpoint(
        decomposed,
        checkpoint_plan,
    )
    return alive_with_cp


def _simulate_peak_with_recompute_set(
    graph_profiler: Any,
    all_activations: Set[Any],
    recompute_set: Set[Any],
    first_bw_use: Dict[Any, Any],
    decomposed: Set[Any],
    optimize_region: str,
) -> int:
    """Return simulated peak memory for a proposed recompute set.

    This uses GraphProfiler's checkpoint-aware lifetime model, then measures
    peak in the configured region. For this project, the default is the forward
    activation-bearing region because optimizer/gradient peaks can otherwise
    hide checkpointing's effect.
    """
    alive_with_cp = _build_alive_with_recompute_set(
        graph_profiler,
        all_activations,
        recompute_set,
        first_bw_use,
        decomposed,
    )
    return _compute_peak_from_alive(graph_profiler, alive_with_cp, optimize_region)


def _select_recompute_nodes(
    graph_profiler: Any,
    candidates: List[CandidateRow],
    retained: Set[Any],
    recompute: Set[Any],
    first_bw_use: Dict[Any, Any],
    required_inputs_map: Dict[Any, Set[Any]],
    config: PolicyConfig,
) -> Tuple[int, float, int, int]:
    """Choose recompute set with a budget-driven greedy policy.

    The loop keeps selecting the candidate with best marginal
    ``delta_peak / recompute_time_ms`` until the configured memory budget is
    met or no remaining candidate can reduce the simulated peak.
    """
    print(
        f"[mu_two_policy] analyzing baseline memory: candidates={len(candidates)}",
        flush=True,
    )
    decomposed = graph_profiler._find_decomposed_parents()
    baseline_alive = graph_profiler._build_alive_ranges(decomposed)
    estimated_peak_before = _compute_peak_from_alive(
        graph_profiler,
        baseline_alive,
        config.optimize_region,
    )
    baseline_overall_peak = _compute_peak_from_alive(
        graph_profiler,
        baseline_alive,
        "overall",
    )
    memory_limit_bytes = _resolve_memory_limit_bytes(estimated_peak_before, config)
    if not candidates:
        return 0, 0.0, estimated_peak_before, memory_limit_bytes
    print(
        f"[mu_two_policy] baseline computed; starting selection loop",
        flush=True,
    )

    print(
        f"[mu_two_policy] selecting recompute nodes: candidates={len(candidates)} "
        f"memory_limit={_fmt_bytes(memory_limit_bytes)} "
        f"baseline_peak={_fmt_bytes(estimated_peak_before)}",
        flush=True,
    )

    # Large graphs can make the greedy search explode because every trial
    # rebuilds checkpoint-aware lifetimes and re-sweeps memory. Keep the small
    # graph behavior intact, but tighten the search budget once the activation
    # set grows beyond a practical threshold.
    large_graph = len(candidates) > 256
    effective_min_candidate_mem_bytes = max(
        int(config.min_candidate_mem_bytes),
        4 * 1024 * 1024 if large_graph else 0,
    )
    effective_max_simulation_candidates = config.max_simulation_candidates
    effective_max_selection_steps = config.max_selection_steps
    if large_graph:
        if effective_max_simulation_candidates <= 0:
            effective_max_simulation_candidates = 16
        else:
            effective_max_simulation_candidates = min(effective_max_simulation_candidates, 16)
        if effective_max_selection_steps <= 0:
            effective_max_selection_steps = 4
        else:
            effective_max_selection_steps = min(effective_max_selection_steps, 4)

    usable = [
        row
        for row in candidates
        if row.mem >= effective_min_candidate_mem_bytes
        and not _is_param_only_candidate(graph_profiler, row)
        and not _is_alias_candidate(graph_profiler, row)
    ]
    if not usable:
        return 0, 0.0, estimated_peak_before, memory_limit_bytes

    ordered = sorted(
        usable,
        key=lambda row: (
            row.mem / max(row.recompute_time_ms, 1e-6),
            row.mem,
            -graph_profiler.node_index.get(row.node, 0),
        ),
        reverse=True,
    )
    if effective_max_simulation_candidates > 0:
        ordered = ordered[: effective_max_simulation_candidates]

    all_activations = set(retained)
    remaining: List[CandidateRow] = list(ordered)

    if large_graph:
        print(
            f"[mu_two_policy] large-graph mode active: usable={len(usable)} "
            f"sim_candidates={len(remaining)} max_steps={effective_max_selection_steps}",
            flush=True,
        )

    current_peak = int(estimated_peak_before)
    selected_recompute_ms = 0.0
    selected_peak_drop = 0
    selected_steps = 0

    while (
        current_peak > memory_limit_bytes
        and remaining
        and (effective_max_selection_steps <= 0 or selected_steps < effective_max_selection_steps)
    ):
        best_row: Optional[CandidateRow] = None
        best_trial_peak = current_peak
        best_trial_first_bw: Dict[Any, Any] = {}
        best_trial_required_inputs: Set[Any] = set()
        best_trial_recompute_ms = 0.0
        best_delta_peak = 0
        best_utility = float("-inf")

        for row in remaining:
            trial_recompute = set(recompute)
            trial_recompute.add(row.node)
            trial_required_inputs, trial_recompute_ms = _estimate_recompute_metrics(
                row.node,
                graph_profiler,
                trial_recompute,
            )

            trial_first_bw = dict(first_bw_use)
            if row.fbw_node is not None:
                trial_first_bw[row.node] = row.fbw_node

            trial_alive = _build_alive_with_recompute_set(
                graph_profiler=graph_profiler,
                all_activations=all_activations,
                recompute_set=trial_recompute,
                first_bw_use=trial_first_bw,
                decomposed=decomposed,
            )
            trial_peak = _compute_peak_from_alive(
                graph_profiler,
                trial_alive,
                config.optimize_region,
            )
            trial_overall_peak = _compute_peak_from_alive(
                graph_profiler,
                trial_alive,
                "overall",
            )
            if config.reject_overall_peak_increase and trial_overall_peak > baseline_overall_peak:
                continue

            delta_peak = current_peak - trial_peak
            if delta_peak <= 0:
                continue

            utility = float(delta_peak) / max(trial_recompute_ms, 1e-6)
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
                best_trial_required_inputs = trial_required_inputs
                best_trial_recompute_ms = trial_recompute_ms
                best_delta_peak = delta_peak
                best_utility = utility

        if best_row is None:
            break

        print(
            f"[mu_two_policy] step {selected_steps + 1}: selected {best_row.node.name} "
            f"delta={_fmt_bytes(best_delta_peak)} "
            f"peak={_fmt_bytes(best_trial_peak)}",
            flush=True,
        )
        recompute.add(best_row.node)
        retained.discard(best_row.node)
        selected_recompute_ms += best_trial_recompute_ms

        if best_row.fbw_node is not None:
            first_bw_use.clear()
            first_bw_use.update(best_trial_first_bw)
        required_inputs_map[best_row.node] = set(best_trial_required_inputs)

        current_peak = best_trial_peak
        selected_peak_drop = int(estimated_peak_before) - int(current_peak)
        remaining = [row for row in remaining if row.node is not best_row.node]
        selected_steps += 1

    if recompute:
        selected_recompute_ms = 0.0
        required_inputs_map.clear()
        for node in sorted(recompute, key=lambda n: graph_profiler.node_index.get(n, 10**9)):
            req_inputs, recompute_time_ms = _estimate_recompute_metrics(
                node,
                graph_profiler,
                recompute,
            )
            required_inputs_map[node] = req_inputs
            selected_recompute_ms += recompute_time_ms

    print(
        f"[mu_two_policy] selection complete: recompute={len(recompute)} "
        f"saved={_fmt_bytes(selected_peak_drop)} final_peak={_fmt_bytes(current_peak)}",
        flush=True,
    )

    return max(0, selected_peak_drop), selected_recompute_ms, int(current_peak), memory_limit_bytes


def _estimate_recompute_metrics(
    target_node: Any,
    graph_profiler: Any,
    recompute_set: Optional[Set[Any]] = None,
) -> Tuple[Set[Any], float]:
    """Estimate the recompute boundary inputs and total runtime cost for one activation."""

    recompute_set = set(recompute_set or {target_node})
    required_inputs: Set[Any] = set()
    recompute_nodes: Set[Any] = set()
    visited: Set[Any] = set()

    def is_retained_activation_boundary(node: Any) -> bool:
        return (
            node not in recompute_set
            and graph_profiler.node_type.get(node) == NodeType.ACT
            and graph_profiler.first_bw_access.get(node) is not None
        )

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

        if is_retained_activation_boundary(node):
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
    candidate_rows = _collect_policy_candidates(graph_profiler, activations)
    excluded_param_only = sum(
        1
        for row in candidate_rows
        if row.mem >= min_candidate_mem and _is_param_only_candidate(graph_profiler, row)
    )
    excluded_alias = sum(
        1
        for row in candidate_rows
        if row.mem >= min_candidate_mem and _is_alias_candidate(graph_profiler, row)
    )
    eligible = [
        row.node
        for row in candidate_rows
        if row.mem >= min_candidate_mem
        and not _is_param_only_candidate(graph_profiler, row)
        and not _is_alias_candidate(graph_profiler, row)
    ]
    largest_candidate_rows = [
        row
        for row in candidate_rows
        if not _is_param_only_candidate(graph_profiler, row)
        and not _is_alias_candidate(graph_profiler, row)
    ]
    largest_candidates = [
        {
            "name": row.node.name,
            "memory_bytes": int(graph_profiler.node_mem_bytes.get(row.node.name, 0)),
            "first_backward_use": (
                graph_profiler.first_bw_access[row.node].name
                if graph_profiler.first_bw_access.get(row.node) is not None
                else None
            ),
        }
        for row in sorted(
            largest_candidate_rows,
            key=lambda candidate: int(graph_profiler.node_mem_bytes.get(candidate.node.name, 0)),
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

        if _is_alias_candidate(
            graph_profiler,
            CandidateRow(node=node, mem=0, recompute_time_ms=0.0, fbw_node=None, req_inputs=set()),
        ):
            errors.append(f"Recompute target {node.name} is an alias/view metadata node")

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

    optimize_region = getattr(plan, "optimize_region", config.optimize_region)
    peak_before = int(plan.estimated_peak_before_bytes or 0)
    peak_after = int(plan.estimated_peak_after_bytes or 0)
    memory_limit = plan.memory_limit_bytes
    memory_budget_satisfied = (
        None if memory_limit is None else peak_after <= int(memory_limit)
    )
    estimated_peak_delta = max(0, peak_before - peak_after)

    decomposed = graph_profiler._find_decomposed_parents()
    baseline_alive = graph_profiler._build_alive_ranges(decomposed)
    plan_alive = graph_profiler._build_alive_ranges_with_checkpoint(decomposed, plan)
    forward_peak_before = _compute_peak_from_alive(graph_profiler, baseline_alive, "forward")
    forward_peak_after = _compute_peak_from_alive(graph_profiler, plan_alive, "forward")
    overall_peak_before = _compute_peak_from_alive(graph_profiler, baseline_alive, "overall")
    overall_peak_after = _compute_peak_from_alive(graph_profiler, plan_alive, "overall")

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

    if overall_peak_after > overall_peak_before:
        warnings.append(
            "Plan reduces the selected region but increases overall peak memory"
        )

    if memory_budget_satisfied is False:
        warnings.append(
            "Policy could not reach the configured memory budget with the eligible candidates"
        )

    param_only_recompute = [
        node.name
        for node in recompute
        if all(
            graph_profiler.node_type.get(inp) == NodeType.PARAM
            for inp in plan.required_recompute_inputs.get(node, set())
        )
    ]
    if param_only_recompute:
        warnings.append(
            "Plan selected parameter-only recompute targets: "
            + ", ".join(sorted(param_only_recompute)[:10])
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
        excluded_param_only_candidates=excluded_param_only,
        excluded_alias_candidates=excluded_alias,
        optimize_region=optimize_region,
        estimated_peak_delta_bytes=estimated_peak_delta,
        memory_limit_bytes=memory_limit,
        memory_budget_satisfied=memory_budget_satisfied,
        forward_peak_before_bytes=forward_peak_before,
        forward_peak_after_bytes=forward_peak_after,
        overall_peak_before_bytes=overall_peak_before,
        overall_peak_after_bytes=overall_peak_after,
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

    print(
        f"[mu_two_policy] building checkpoint plan: region={config.optimize_region} "
        f"budget={config.default_memory_budget_fraction:.2f}",
        flush=True,
    )

    activations = [
        n
        for n in graph_profiler.intermediate_nodes
        if graph_profiler.first_bw_access.get(n) is not None
    ]
    retained: Set[Any] = set(activations)
    recompute: Set[Any] = set()
    first_bw_use: Dict[Any, Any] = {}
    required_inputs_map: Dict[Any, Set[Any]] = {}

    candidates = _collect_policy_candidates(graph_profiler, activations)
    print(
        f"[mu_two_policy] candidate activations={len(activations)} "
        f"eligible_for_search={len(candidates)}",
        flush=True,
    )
    (
        selected_mem_bytes,
        selected_recompute_ms,
        estimated_peak_after,
        memory_limit_bytes,
    ) = _select_recompute_nodes(
        graph_profiler=graph_profiler,
        candidates=candidates,
        retained=retained,
        recompute=recompute,
        first_bw_use=first_bw_use,
        required_inputs_map=required_inputs_map,
        config=config,
    )
    estimated_peak_before = estimated_peak_after + selected_mem_bytes

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
        optimize_region=config.optimize_region,
    )
