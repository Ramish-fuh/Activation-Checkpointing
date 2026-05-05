#!/usr/bin/env python3
"""
Diagnostic script to analyze grad node behavior and checkpoint memory reduction.

This script runs profile diagnostics for Bert at batch sizes 4 and 8, printing:
1. sep_idx and sep_bw_idx
2. Number of grad nodes and their names/indices
3. Per-step grad memory around backward start
4. Checkpoint plan stats
5. Top OTHER contributors before/after checkpoint
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import copy
from benchmarks import Experiment
from graph_prof import GraphProfiler, NodeType
from mu_two_policy import PolicyConfig, build_checkpoint_plan, validate_checkpoint_plan
from activation_checkpoint import apply_checkpoint_plan, smoke_check_graph
import torch.utils._pytree as pytree


def run_diagnostic(batch_size: int):
    print(f"\n{'='*80}")
    print(f"DIAGNOSTICS: Bert batch_size={batch_size}")
    print(f"{'='*80}\n")
    
    # Initialize experiment
    experiment = Experiment("Bert", batch_size)
    experiment.init_opt_states()
    
    # Compile
    from graph_tracer import _compile
    compiled = _compile(
        experiment.train_step,
        experiment.model,
        experiment.optimizer,
        experiment.example_inputs,
    )
    baseline_gm = copy.deepcopy(compiled.gm)
    flat_inputs = compiled.flat_state + pytree.tree_flatten(
        [(experiment.model, experiment.optimizer, experiment.example_inputs), {}]
    )[0]
    
    # Profile baseline
    profiler = GraphProfiler(compiled.gm)
    with torch.no_grad():
        profiler.run(*flat_inputs)
        profiler.reset_stats()
        profiler.run(*flat_inputs)
    profiler.aggregate_stats()
    
    # ========== PART 1: Separator indices and grad nodes ==========
    print("--- Separator Indices ---")
    print(f"  sep_idx: {profiler.sep_idx} (node: {profiler.sep_node.name})")
    print(f"  sep_bw_idx: {profiler.sep_bw_idx} (node: {profiler.sep_backward_node.name})")
    
    print("\n--- Grad Nodes ---")
    grad_nodes = sorted(profiler.grad_nodes, key=lambda n: profiler.node_index[n])
    print(f"  Total grad nodes: {len(grad_nodes)}")
    print("\n  Sample grad nodes (first 10):")
    for i, node in enumerate(grad_nodes[:10]):
        idx = profiler.node_index[node]
        mem_bytes = profiler.node_mem_bytes.get(node.name, 0)
        mem_mb = mem_bytes / (1024**2)
        print(f"    {i}. {node.name} (index={idx}, mem={mem_mb:.4f} MB)")
    
    # ========== PART 2: Per-step grad memory around backward start ==========
    print("\n--- Grad Memory Timeline Around Backward Start ---")
    decomposed = profiler._find_decomposed_parents()
    alive_baseline = profiler._build_alive_ranges(decomposed)
    steps, by_type_series, _ = profiler._build_memory_timeline(alive_baseline)
    
    # Print a small window around sep_bw_idx
    window_start = max(profiler.sep_bw_idx - 4, 0)
    window_end = min(profiler.sep_bw_idx + 8, len(steps))
    print(f"  Window: ops {window_start} to {window_end} (sep_bw_idx={profiler.sep_bw_idx})")
    print(f"  {'Op':>4s} {'Grad MB':>12s}")
    for step_idx in range(window_start, window_end):
        series_idx = steps.index(step_idx) if step_idx in steps else None
        if series_idx is not None:
            grad_bytes = by_type_series[series_idx].get(NodeType.GRAD, 0)
            grad_mb = grad_bytes / (1024**2)
            marker = " <-- sep_bw_idx" if step_idx == profiler.sep_bw_idx else ""
            print(f"  {step_idx:>4d} {grad_mb:>12.4f}{marker}")
    
    # ========== PART 3: Checkpoint plan stats ==========
    print("\n--- Checkpoint Plan ---")
    config = PolicyConfig(default_memory_budget_fraction=0.5)
    plan = build_checkpoint_plan(profiler, config)
    report = validate_checkpoint_plan(profiler, plan, config)
    
    print(f"  Retained activations: {len(plan.retained_nodes)}")
    print(f"  Recomputed activations: {len(plan.recompute_nodes)}")
    print(f"  Forward peak before: {report.forward_peak_before_bytes / (1024**2):.2f} MB")
    print(f"  Forward peak after: {report.forward_peak_after_bytes / (1024**2):.2f} MB")
    print(f"  Reduction: {(1 - report.forward_peak_after_bytes / report.forward_peak_before_bytes) * 100:.1f}%")
    
    # Sample recompute nodes
    print(f"\n  Sample recompute nodes (first 5):")
    recompute_sorted = sorted(plan.recompute_nodes, key=lambda n: profiler.node_index[n])
    for i, node in enumerate(recompute_sorted[:5]):
        idx = profiler.node_index[node]
        print(f"    {i}. {node.name} (index={idx})")
    
    # ========== PART 4: Top OTHER contributors before/after checkpoint ==========
    print("\n--- OTHER Memory Contributors At Forward Peak ---")
    
    # Find forward peak step in baseline
    fw_peak_step, fw_peak_bytes, fw_breakdown = profiler._sweep_for_forward_peak(alive_baseline)
    print(f"  Forward peak occurs at step: {fw_peak_step} ({fw_peak_bytes / (1024**2):.2f} MB)")
    print(f"\n  Before checkpoint (top OTHER contributors at forward peak):")
    
    # Collect OTHER nodes alive at forward peak in baseline
    other_contributions_baseline = {}
    for node, (born, dies) in alive_baseline.items():
        if born <= fw_peak_step <= dies and profiler.node_type.get(node) == NodeType.OTHER:
            mem_bytes = profiler.node_mem_bytes.get(node.name, 0)
            other_contributions_baseline[node.name] = mem_bytes
    
    # Sort and print top OTHER
    sorted_other_baseline = sorted(other_contributions_baseline.items(), key=lambda x: -x[1])[:10]
    for name, mem_bytes in sorted_other_baseline:
        mem_mb = mem_bytes / (1024**2)
        print(f"    {name}: {mem_mb:.4f} MB")
    
    total_other_baseline = sum(other_contributions_baseline.values())
    print(f"  Total OTHER at forward peak (baseline): {total_other_baseline / (1024**2):.2f} MB")
    
    # Now build modeled alive ranges with checkpoint
    alive_with_checkpoint = profiler._build_alive_ranges_with_checkpoint(plan)
    fw_peak_step_ckpt, fw_peak_bytes_ckpt, fw_breakdown_ckpt = profiler._sweep_for_forward_peak(alive_with_checkpoint)
    
    print(f"\n  After checkpoint (top OTHER contributors at forward peak):")
    print(f"  Forward peak occurs at step: {fw_peak_step_ckpt} ({fw_peak_bytes_ckpt / (1024**2):.2f} MB)")
    
    # Collect OTHER nodes alive at forward peak with checkpoint
    other_contributions_ckpt = {}
    for node, (born, dies) in alive_with_checkpoint.items():
        if born <= fw_peak_step_ckpt <= dies and profiler.node_type.get(node) == NodeType.OTHER:
            mem_bytes = profiler.node_mem_bytes.get(node.name, 0)
            other_contributions_ckpt[node.name] = mem_bytes
    
    sorted_other_ckpt = sorted(other_contributions_ckpt.items(), key=lambda x: -x[1])[:10]
    for name, mem_bytes in sorted_other_ckpt:
        mem_mb = mem_bytes / (1024**2)
        print(f"    {name}: {mem_mb:.4f} MB")
    
    total_other_ckpt = sum(other_contributions_ckpt.values())
    print(f"  Total OTHER at forward peak (checkpoint): {total_other_ckpt / (1024**2):.2f} MB")
    
    other_reduction = (total_other_baseline - total_other_ckpt) / max(total_other_baseline, 1e-6)
    print(f"\n  OTHER reduction with checkpoint: {other_reduction * 100:.1f}%")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Run for both batch sizes
    for bs in [4, 8]:
        try:
            run_diagnostic(bs)
        except Exception as e:
            print(f"Error running diagnostic for bs={bs}: {e}")
            import traceback
            traceback.print_exc()
