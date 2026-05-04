import copy
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import torch.utils._pytree as pytree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activation_checkpoint import apply_checkpoint_plan, smoke_check_graph
from benchmarks import Experiment
from graph_prof import GraphProfiler
from graph_tracer import _compile
from mu_two_policy import PolicyConfig, build_checkpoint_plan, validate_checkpoint_plan


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase 3 rewrite tests require CUDA"
)


def _build_checkpointed_graph(batch_size: int):
    experiment = Experiment("Bert", batch_size)
    experiment.init_opt_states()

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

    profiler = GraphProfiler(compiled.gm)
    with torch.no_grad():
        profiler.run(*flat_inputs)
        profiler.reset_stats()
        profiler.run(*flat_inputs)
    profiler.aggregate_stats()

    config = PolicyConfig(default_memory_budget_fraction=0.5)
    plan = build_checkpoint_plan(profiler, config)
    report = validate_checkpoint_plan(profiler, plan, config)

    assert report.ok, report.format_summary()
    assert plan.recompute_nodes, "Expected at least one activation to be recomputed"

    checkpoint_gm = apply_checkpoint_plan(copy.deepcopy(compiled.gm), plan)
    ok, message = smoke_check_graph(checkpoint_gm, flat_inputs, reference_gm=baseline_gm)
    assert ok, message

    return plan, report


@lru_cache(maxsize=None)
def _cached_checkpointed_graph(batch_size: int):
    return _build_checkpointed_graph(batch_size)


@pytest.mark.parametrize("batch_size", [4, 8])
def test_phase3_checkpoint_rewrite_bert(batch_size: int):
    plan, report = _cached_checkpointed_graph(batch_size)

    assert len(plan.recompute_nodes) > 0
    assert report.checkpointable_activations > 0
    assert report.eligible_candidates > 0
    assert report.forward_peak_after_bytes <= report.forward_peak_before_bytes


@pytest.mark.parametrize("batch_size", [4, 8])
def test_phase3_checkpoint_plan_is_deterministic_bert(batch_size: int):
    config = PolicyConfig(default_memory_budget_fraction=0.5)
    plan1, _ = _cached_checkpointed_graph(batch_size)
    plan2, _ = _cached_checkpointed_graph(batch_size)

    assert {node.name for node in plan1.recompute_nodes} == {
        node.name for node in plan2.recompute_nodes
    }
