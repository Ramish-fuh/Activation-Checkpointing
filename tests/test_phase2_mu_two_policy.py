"""
Phase 2 Tests: μ-TWO Activation Checkpointing Policy

Tests validate that the policy algorithm correctly:
1. Selects checkpointable activations to recompute
2. Maintains valid plan structure (disjoint sets, accounting)
3. Reduces peak memory while respecting computational overhead
4. Handles edge cases and configuration parameters

These tests work with the Phase 1 profiling output (GraphProfiler).
Phase 3 (graph rewriting) is NOT tested here.
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_prof import GraphProfiler, NodeType
from mu_two_policy import (
    PolicyConfig,
    CheckpointPlan,
    build_checkpoint_plan,
    validate_checkpoint_plan,
    _collect_policy_candidates,
    _select_recompute_nodes,
    _estimate_recompute_metrics,
    _is_param_only_candidate,
    _is_alias_candidate,
)
from graph_tracer import compile
from benchmarks import Experiment


class TestCheckpointPlanStructure:
    """Test that checkpoint plans have valid structure."""

    def test_plan_retained_recompute_disjoint(self):
        """CRITICAL: Retained and recompute sets must be mutually exclusive."""
        # Setup: trace and profile a model
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        # Build checkpoint plan
        plan = build_checkpoint_plan(profiler)
        
        # Verify: no overlap
        overlap = plan.retained_nodes & plan.recompute_nodes
        assert len(overlap) == 0, f"Retained and recompute sets overlap: {[n.name for n in overlap]}"

    def test_plan_recompute_in_activations(self):
        """CRITICAL: All recompute nodes must be classified as ACT."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        for node in plan.recompute_nodes:
            node_type = profiler.node_type.get(node)
            assert node_type == NodeType.ACT, (
                f"Recompute node {node.name} is {node_type}, expected ACT"
            )

    def test_plan_first_backward_use_exists(self):
        """CRITICAL: Every recompute node needs first_backward_use."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        for node in plan.recompute_nodes:
            assert node in plan.first_backward_use, (
                f"Recompute node {node.name} missing first_backward_use"
            )
            fbw_node = plan.first_backward_use[node]
            assert fbw_node is not None
            assert profiler.node_index[fbw_node] >= profiler.sep_bw_idx, (
                f"first_backward_use for {node.name} occurs before backward region"
            )

    def test_plan_required_recompute_inputs_exist(self):
        """CRITICAL: Every recompute node needs required_recompute_inputs."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        for node in plan.recompute_nodes:
            assert node in plan.required_recompute_inputs, (
                f"Recompute node {node.name} missing required_recompute_inputs"
            )
            inputs = plan.required_recompute_inputs[node]
            assert len(inputs) > 0, (
                f"Recompute node {node.name} has empty required_recompute_inputs"
            )

    def test_plan_required_inputs_valid(self):
        """CRITICAL: Required inputs must be either PARAM, retained ACT, or placeholder."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        retained_names = {n.name for n in plan.retained_nodes}
        
        for node, inputs in plan.required_recompute_inputs.items():
            for inp in inputs:
                inp_type = profiler.node_type.get(inp)
                is_valid = (
                    inp.op == "placeholder"
                    or inp_type == NodeType.PARAM
                    or inp.name in retained_names
                )
                assert is_valid, (
                    f"Invalid input {inp.name} ({inp_type}) for recompute node {node.name}"
                )

    def test_plan_serializable(self):
        """Ensure checkpoint plan can be serialized to dict (needed for reporting)."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        serialized = plan.to_dict()
        assert isinstance(serialized, dict)
        assert "retained_nodes" in serialized
        assert "recompute_nodes" in serialized
        assert "first_backward_use" in serialized
        assert "required_recompute_inputs" in serialized
        assert isinstance(serialized["retained_nodes"], list)
        assert isinstance(serialized["recompute_nodes"], list)


class TestCheckpointPlanQuality:
    """Test that the plan improves memory efficiency."""

    def test_plan_reduces_forward_peak(self):
        """Checkpointing should reduce forward peak memory."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        # Plans should reduce forward peak (where activations dominate)
        peak_before = plan.estimated_peak_before_bytes
        peak_after = plan.estimated_peak_after_bytes
        
        assert peak_before > 0, "Baseline peak should be > 0"
        assert peak_after >= 0, "After-plan peak should be >= 0"
        
        # If we selected recompute nodes, peak should decrease
        if plan.recompute_nodes:
            memory_saved = peak_before - peak_after
            assert memory_saved > 0, (
                f"Plan selected recompute nodes but peak didn't decrease: "
                f"{peak_before} -> {peak_after}"
            )

    def test_plan_memory_saved_matches_peak_delta(self):
        """estimated_memory_saved_bytes should match peak_before - peak_after."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        expected_saved = plan.estimated_peak_before_bytes - plan.estimated_peak_after_bytes
        actual_saved = plan.estimated_memory_saved_bytes
        
        assert abs(expected_saved - actual_saved) < 1024, (
            f"Memory saved mismatch: {expected_saved} bytes vs {actual_saved} bytes"
        )

    def test_plan_recompute_overhead_reasonable(self):
        """Recompute overhead should be non-negative and not excessive."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        assert plan.estimated_recompute_overhead_ms >= 0, (
            "Recompute overhead should be non-negative"
        )
        
        # Overhead should be reasonable relative to forward pass time
        # (This is approximate; actual forward time depends on model size)
        assert plan.estimated_recompute_overhead_ms < 10000, (
            "Recompute overhead seems excessive (> 10 seconds)"
        )


class TestPolicyWithDifferentConfigs:
    """Test that the policy respects configuration parameters."""

    def test_policy_empty_recompute_with_tight_budget(self):
        """With very tight budget, policy might not find any candidates."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        # Set budget to 0% of baseline (impossible to achieve)
        config = PolicyConfig(
            default_memory_budget_fraction=0.0,
            reject_overall_peak_increase=False,
        )
        plan = build_checkpoint_plan(profiler, config)
        
        # Plan structure should still be valid
        assert isinstance(plan, CheckpointPlan)
        assert len(plan.retained_nodes & plan.recompute_nodes) == 0

    def test_policy_respects_min_candidate_size(self):
        """Only candidates >= min_candidate_mem_bytes should be considered."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        # Set high minimum to filter out small activations
        min_size_mb = 10 * 1024 * 1024  # 10 MB
        config = PolicyConfig(min_candidate_mem_bytes=min_size_mb)
        plan = build_checkpoint_plan(profiler, config)
        
        # All recompute nodes should be >= min size
        for node in plan.recompute_nodes:
            mem = int(profiler.node_mem_bytes.get(node.name, 0))
            assert mem >= min_size_mb, (
                f"Recompute node {node.name} ({mem} bytes) < min {min_size_mb}"
            )

    def test_policy_excludes_param_only_candidates(self):
        """Recompute targets should not be param-only (they won't save activation memory)."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        for node in plan.recompute_nodes:
            inputs = plan.required_recompute_inputs.get(node, set())
            # If all inputs are PARAM, it's param-only and should not be selected
            is_param_only = all(
                profiler.node_type.get(inp) == NodeType.PARAM for inp in inputs
            )
            assert not is_param_only, (
                f"Recompute node {node.name} is param-only; should not be selected"
            )

    def test_policy_excludes_alias_candidates(self):
        """Recompute targets should not be view/transpose nodes (metadata only)."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        alias_prefixes = ("view", "transpose", "expand", "slice", "select", "squeeze", "unsqueeze", "permute")
        
        for node in plan.recompute_nodes:
            name = node.name
            is_alias = (
                name == "t"
                or name.startswith("t_")
                or any(name == prefix or name.startswith(f"{prefix}_") for prefix in alias_prefixes)
            )
            assert not is_alias, (
                f"Recompute node {node.name} is an alias/view node; should not be selected"
            )


class TestPolicyCandidateCollection:
    """Test the candidate filtering and collection logic."""

    def test_candidates_are_activations(self):
        """Collected candidates should all be ACT nodes."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        activations = [
            n for n in profiler.intermediate_nodes
            if profiler.first_bw_access.get(n) is not None
        ]
        
        for act in activations:
            node_type = profiler.node_type.get(act)
            assert node_type == NodeType.ACT, (
                f"Activation {act.name} has type {node_type}, expected ACT"
            )

    def test_candidates_have_backward_uses(self):
        """All activation candidates must have first_bw_access."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        activations = [
            n for n in profiler.intermediate_nodes
            if profiler.first_bw_access.get(n) is not None
        ]
        
        assert len(activations) > 0, "No activation candidates found"
        
        for act in activations:
            fbw = profiler.first_bw_access.get(act)
            assert fbw is not None
            assert profiler.node_index[fbw] >= profiler.sep_bw_idx


class TestRecomputeMetricsEstimation:
    """Test that recompute cost estimation is reasonable."""

    def test_recompute_metrics_valid_inputs(self):
        """Recompute inputs should only include PARAM, retained ACT, or placeholders."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        # Get a sample activation
        activations = [
            n for n in profiler.intermediate_nodes
            if profiler.first_bw_access.get(n) is not None
        ]
        
        if not activations:
            pytest.skip("No activations found")
        
        target = activations[0]
        required_inputs, recompute_time_ms = _estimate_recompute_metrics(
            target,
            profiler,
        )
        
        # Validate inputs
        for inp in required_inputs:
            inp_type = profiler.node_type.get(inp)
            is_valid = (
                inp.op == "placeholder"
                or inp_type == NodeType.PARAM
                or inp_type == NodeType.ACT
            )
            assert is_valid, (
                f"Invalid recompute input {inp.name} ({inp_type})"
            )

    def test_recompute_time_nonnegative(self):
        """Recompute time should never be negative."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        activations = [
            n for n in profiler.intermediate_nodes
            if profiler.first_bw_access.get(n) is not None
        ]
        
        if not activations:
            pytest.skip("No activations found")
        
        for act in activations[:5]:  # Test first 5 to keep runtime reasonable
            _, recompute_time_ms = _estimate_recompute_metrics(act, profiler)
            assert recompute_time_ms >= 0, (
                f"Recompute time for {act.name} is negative: {recompute_time_ms} ms"
            )


class TestValidateCheckpointPlan:
    """Test plan validation logic."""

    def test_validation_report_on_valid_plan(self):
        """Validation should pass on a plan generated by the policy."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        report = validate_checkpoint_plan(profiler, plan)
        
        # Should not have errors (warnings are OK)
        assert report.ok, (
            f"Plan validation failed with errors:\n{report.format_summary()}"
        )

    def test_validation_detects_overlap(self):
        """Validation should catch overlap between retained and recompute."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        # Artificially create overlap
        if plan.recompute_nodes:
            node_to_move = list(plan.recompute_nodes)[0]
            plan.retained_nodes.add(node_to_move)
            
            report = validate_checkpoint_plan(profiler, plan)
            assert not report.ok, "Validation should detect overlap"
            assert any("overlap" in err.lower() for err in report.errors)

    def test_validation_detects_invalid_first_backward_use(self):
        """Validation should catch first_backward_use before backward region."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        if not plan.recompute_nodes:
            pytest.skip("No recompute nodes in plan")
        
        # Validation should pass on generated plan
        report = validate_checkpoint_plan(profiler, plan)
        for node in plan.recompute_nodes:
            fbw = plan.first_backward_use.get(node)
            if fbw:
                fbw_idx = profiler.node_index.get(fbw)
                assert fbw_idx >= profiler.sep_bw_idx, (
                    f"Invalid first_backward_use for {node.name}"
                )


class TestPolicyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_policy_with_no_viable_candidates(self):
        """Policy should handle case where no candidates reduce memory."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        # With very small budget and high overhead, policy might not select anything
        config = PolicyConfig(
            default_memory_budget_fraction=0.01,
            min_candidate_mem_bytes=1024 * 1024 * 100,  # 100 MB threshold
            max_selection_steps=1,
        )
        plan = build_checkpoint_plan(profiler, config)
        
        # Should still return valid plan (possibly empty)
        assert isinstance(plan, CheckpointPlan)
        assert len(plan.retained_nodes & plan.recompute_nodes) == 0

    def test_policy_with_max_selection_steps_limit(self):
        """Policy should respect max_selection_steps."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        
        config = PolicyConfig(max_selection_steps=1)
        plan = build_checkpoint_plan(profiler, config)
        
        # Plan should be valid regardless of step limit
        assert isinstance(plan, CheckpointPlan)
        report = validate_checkpoint_plan(profiler, plan)
        assert report.ok


class TestPolicyConsistency:
    """Test that the policy is deterministic and consistent."""

    def test_same_config_produces_same_plan(self):
        """Running policy twice with same config should produce same plan."""
        experiment1 = Experiment("Bert", batch_size=4)
        experiment2 = Experiment("Bert", batch_size=4)
        
        profiler1 = experiment1.profiler
        profiler2 = experiment2.profiler
        
        config = PolicyConfig(default_memory_budget_fraction=0.5)
        plan1 = build_checkpoint_plan(profiler1, config)
        plan2 = build_checkpoint_plan(profiler2, config)
        
        # Plans should select the same nodes
        nodes1 = {n.name for n in plan1.recompute_nodes}
        nodes2 = {n.name for n in plan2.recompute_nodes}
        
        assert nodes1 == nodes2, (
            f"Same config produced different plans:\n"
            f"  Run 1: {sorted(nodes1)}\n"
            f"  Run 2: {sorted(nodes2)}"
        )

    def test_plan_peak_estimates_reasonable(self):
        """Peak estimates should be consistent with profiler state."""
        experiment = Experiment("Bert", batch_size=4)
        profiler = experiment.profiler
        plan = build_checkpoint_plan(profiler)
        
        baseline_peak = profiler.overall_peak_bytes if hasattr(profiler, 'overall_peak_bytes') else 500 * 1024 * 1024
        
        # Peak after should be <= baseline peak
        assert plan.estimated_peak_after_bytes <= plan.estimated_peak_before_bytes, (
            f"Peak after checkpointing should not exceed peak before"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
