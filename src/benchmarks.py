# ============================================================================
# benchmarks.py
#
# Experiment harness for tracing and profiling a single training step.
# Supports BERT and Resnet152 architectures. The
# graph_transformation callback runs the GraphProfiler and saves a
# peak-memory breakdown plot after the first compiled iteration.
# ============================================================================

import os
import sys
import json
import copy
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torchvision.models import resnet152
from graph_prof import GraphProfiler
from mu_two_policy import build_checkpoint_plan, PolicyConfig, validate_checkpoint_plan
from activation_checkpoint import apply_checkpoint_plan, clone_graph_inputs, smoke_check_graph
from graph_tracer import SEPFunction, compile

try:
    from transformers import BertConfig, BertForMaskedLM
except ImportError:
    BertConfig = None
    BertForMaskedLM = None


model_names: List[str] = [
    "Bert",
    "Resnet152",
]

model_batch_sizes: Dict[str, int] = {
    "Bert": 4,
    "Resnet152": 4,
}


class Experiment:
    """Encapsulates model creation, training-step definition, and the
    graph-transformation callback used by compile() to profile a single
    training iteration."""

    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        """Build the model, example inputs, optimizer, and training-step
        closure for the requested architecture."""
        assert model_name in model_names, f"Model {model_name} not found in model names {model_names}"
        dev = torch.device("cuda")
        self.model_name = model_name
        self.batch_size = batch_size

        if self.model_name == "Bert":
            if BertConfig is None or BertForMaskedLM is None:
                raise ImportError(
                    "BERT experiment requires transformers. Install with: pip install transformers"
                )

            vocab_size = 30522
            bsz, seq_len = self.batch_size, 128
            with torch.device(dev):
                bert_cfg = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=4,
                    intermediate_size=1024,
                    max_position_embeddings=512,
                )
                self.model = BertForMaskedLM(bert_cfg)

            input_ids = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            labels = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            self.example_inputs = (input_ids, labels)

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                out = model(input_ids=example_inputs[0], labels=example_inputs[1])
                loss = SEPFunction.apply(out.loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, fused=True, capturable=True)

        elif self.model_name == "Resnet152":
            inp = torch.randn(self.batch_size, 3, 224, 224, device=dev)
            num_classes = 10
            target = torch.randint(0, num_classes, (self.batch_size,), device=dev)
            self.example_inputs = (inp, target)
            with torch.device(dev):
                self.model = resnet152(num_classes=num_classes)

            def resnet_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = self.loss_fn(model(example_inputs[0]), example_inputs[1])
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2, fused=True, capturable=True)
            self.train_step = resnet_train_step

    def _time_graph_once_ms(
        self,
        gm: fx.GraphModule,
        args: Any,
    ) -> float:
        """Time one FX graph replay on freshly cloned inputs."""
        bench_args = clone_graph_inputs(args)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gm(*bench_args)
                end.record()
                torch.cuda.synchronize()
                return float(start.elapsed_time(end))

            start_time = time.perf_counter()
            gm(*bench_args)
            return float((time.perf_counter() - start_time) * 1000.0)

    def _measure_latency_comparison(
        self,
        baseline_gm: fx.GraphModule,
        checkpoint_gm: fx.GraphModule,
        args: Any,
        warmup_iters: int = 2,
        measured_iters: int = 5,
    ) -> Dict[str, Any]:
        """Measure baseline/checkpoint graph replay latency in paired trials."""
        measured_iters = max(1, measured_iters)
        warmup_iters = max(0, warmup_iters)

        with torch.no_grad():
            for _ in range(warmup_iters):
                baseline_gm(*clone_graph_inputs(args))
                checkpoint_gm(*clone_graph_inputs(args))

        baseline_samples: List[float] = []
        checkpoint_samples: List[float] = []
        for idx in range(measured_iters):
            if idx % 2 == 0:
                baseline_samples.append(self._time_graph_once_ms(baseline_gm, args))
                checkpoint_samples.append(self._time_graph_once_ms(checkpoint_gm, args))
            else:
                checkpoint_samples.append(self._time_graph_once_ms(checkpoint_gm, args))
                baseline_samples.append(self._time_graph_once_ms(baseline_gm, args))

        baseline_ms = sum(baseline_samples) / len(baseline_samples)
        checkpoint_ms = sum(checkpoint_samples) / len(checkpoint_samples)
        overhead_ms = checkpoint_ms - baseline_ms
        overhead_percent = (checkpoint_ms / baseline_ms - 1.0) * 100.0 if baseline_ms else None
        return {
            "kind": "paired_raw_fx_graph_replay_ms_on_fresh_cloned_inputs",
            "warmup_iters": warmup_iters,
            "measured_iters": measured_iters,
            "baseline_ms": baseline_ms,
            "checkpoint_ms": checkpoint_ms,
            "overhead_ms": overhead_ms,
            "overhead_percent": overhead_percent,
            "baseline_samples_ms": baseline_samples,
            "checkpoint_samples_ms": checkpoint_samples,
        }

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        """Cross-entropy loss, flattened over all sequence positions."""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def init_opt_states(self):
        """Run one dummy optimizer step so Adam's momentum/variance buffers
        are allocated before tracing (make_fx requires them to exist)."""
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def graph_transformation(self, gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        """Profile the traced graph and save a peak-memory breakdown plot.

        Called once by compile() after the first tracing iteration.  Runs
        warm-up iterations to stabilise CUDA caches, then profiles several
        iterations, prints per-node stats, and writes a bar chart to the
        ``plots/`` directory.
        """
        print(gm.graph.print_tabular())
        warm_up_iters, profile_iters = 2, 3
        graph_profiler = GraphProfiler(gm)

        with torch.no_grad():
            for _ in range(warm_up_iters):
                graph_profiler.run(*args)
            graph_profiler.reset_stats()

            for _ in range(profile_iters):
                graph_profiler.run(*args)
            graph_profiler.aggregate_stats()
            graph_profiler.print_stats()

            # Let policy select checkpoint set automatically (no hard budget).
            policy_config = PolicyConfig()
            plan = build_checkpoint_plan(graph_profiler, policy_config)
            validation = validate_checkpoint_plan(graph_profiler, plan, policy_config)
            rewrite_status = {
                "attempted": False,
                "applied": False,
                "error": "",
                "reason": "",
            }

            print(
                "Checkpoint plan summary: "
                f"region={plan.optimize_region}, "
                f"recompute_nodes={len(plan.recompute_nodes)}, "
                f"saved={plan.estimated_memory_saved_bytes / 1024**2:.1f} MB, "
                f"peak_before={plan.estimated_peak_before_bytes / 1024**2:.1f} MB, "
                f"peak_after={plan.estimated_peak_after_bytes / 1024**2:.1f} MB, "
                f"memory_limit={plan.memory_limit_bytes / 1024**2:.1f} MB"
            )
            print(validation.format_summary())

            plots_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'plots'
            )
            os.makedirs(plots_dir, exist_ok=True)

            plan_path = os.path.join(
                plots_dir,
                f"checkpoint_plan_{self.model_name}_bs{self.batch_size}.json",
            )
            report_path = os.path.join(
                plots_dir,
                f"experiment_summary_{self.model_name}_bs{self.batch_size}.json",
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                plan_payload = plan.to_dict()
                plan_payload["validation"] = validation.to_dict()
                plan_payload["rewrite_status"] = rewrite_status
                json.dump(plan_payload, f, indent=2)
            print(f"Saved plan: {os.path.abspath(plan_path)}")

            # Generate forward-only peak plot (with checkpoint comparison)
            save_path_fw = os.path.join(
                plots_dir,
                f"peak_memory_breakdown_fw_{self.model_name}_bs{self.batch_size}.png",
            )
            try:
                graph_profiler.plot_peak_memory_breakdown(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_fw,
                    forward_only=True,
                    checkpoint_plan=plan,
                )
            except Exception as e:
                print(f"Warning: could not save forward-only plot: {e}")

            # Generate overall peak plot (with checkpoint comparison)
            save_path_overall = os.path.join(
                plots_dir,
                f"peak_memory_breakdown_overall_{self.model_name}_bs{self.batch_size}.png",
            )
            try:
                graph_profiler.plot_peak_memory_breakdown(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_overall,
                    forward_only=False,
                    checkpoint_plan=plan,
                )
            except Exception as e:
                print(f"Warning: could not save overall plot: {e}")

            # Generate memory-over-time timeline against op id
            save_path_timeline = os.path.join(
                plots_dir,
                f"memory_vs_opid_{self.model_name}_bs{self.batch_size}.png",
            )
            try:
                graph_profiler.plot_memory_vs_opid(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_timeline,
                    checkpoint_plan=plan,
                )
            except Exception as e:
                print(f"Warning: could not save memory-vs-opid plot: {e}")

            # Generate phase-wise peak memory (forward/loss/backward)
            save_path_phase = os.path.join(
                plots_dir,
                f"memory_by_phase_{self.model_name}_bs{self.batch_size}.png",
            )
            try:
                graph_profiler.plot_phase_memory_summary(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_phase,
                    checkpoint_plan=plan,
                )
            except Exception as e:
                print(f"Warning: could not save phase-memory plot: {e}")

            # Generate component timeline (weights / gradients / feature maps)
            save_path_components = os.path.join(
                plots_dir,
                f"memory_components_{self.model_name}_bs{self.batch_size}.png",
            )
            try:
                graph_profiler.plot_memory_components_vs_opid(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_components,
                    checkpoint_plan=None,
                )
            except Exception as e:
                print(f"Warning: could not save component-memory plot: {e}")

            save_path_components_cp = os.path.join(
                plots_dir,
                f"memory_components_{self.model_name}_bs{self.batch_size}_with_checkpoint.png",
            )
            try:
                graph_profiler.plot_memory_components_vs_opid(
                    title=f"{self.model_name} (bs={self.batch_size})",
                    save_path=save_path_components_cp,
                    checkpoint_plan=plan,
                )
            except Exception as e:
                print(f"Warning: could not save component-memory checkpoint plot: {e}")

            latency_report: Dict[str, Any] = {
                "kind": "paired_raw_fx_graph_replay_ms_on_fresh_cloned_inputs",
                "warmup_iters": 2,
                "measured_iters": 5,
                "baseline_ms": None,
                "checkpoint_ms": None,
                "overhead_ms": None,
                "overhead_percent": None,
            }

            if not validation.ok:
                rewrite_status["reason"] = "validation failed"
            elif not plan.recompute_nodes:
                rewrite_status["reason"] = "plan selected no recompute nodes"
            else:
                rewrite_status["attempted"] = True
                try:
                    candidate_gm = copy.deepcopy(gm)
                    candidate_gm = apply_checkpoint_plan(candidate_gm, plan)
                    ok, error = smoke_check_graph(candidate_gm, args, reference_gm=gm)
                    if ok:
                        baseline_gm = gm
                        gm = candidate_gm
                        rewrite_status["applied"] = True
                        rewrite_status["reason"] = "rewrite smoke check passed"
                        try:
                            latency_report = self._measure_latency_comparison(
                                baseline_gm,
                                candidate_gm,
                                args,
                            )
                        except Exception as e:
                            latency_report["comparison_error"] = repr(e)
                            print(f"Warning: could not measure graph latency comparison: {e}")
                        print("Checkpoint rewrite applied to FX graph.")
                    else:
                        rewrite_status["error"] = error
                        rewrite_status["reason"] = "rewrite smoke check failed"
                        print(
                            "Warning: checkpoint rewrite failed smoke check; "
                            f"returning original graph. Error: {error}"
                        )
                except Exception as e:
                    rewrite_status["error"] = repr(e)
                    rewrite_status["reason"] = "rewrite raised exception"
                    print(
                        "Warning: checkpoint rewrite raised an exception; "
                        f"returning original graph. Error: {e}"
                    )

            with open(plan_path, "w", encoding="utf-8") as f:
                plan_payload = plan.to_dict()
                plan_payload["validation"] = validation.to_dict()
                plan_payload["rewrite_status"] = rewrite_status
                json.dump(plan_payload, f, indent=2)

            experiment_summary = {
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "policy": {
                    "name": "single_model_budgeted_greedy_mu_two_style",
                    "optimize_region": plan.optimize_region,
                    "memory_limit_bytes": plan.memory_limit_bytes,
                    "memory_budget_satisfied": validation.memory_budget_satisfied,
                    "recompute_nodes": sorted(n.name for n in plan.recompute_nodes),
                    "estimated_recompute_overhead_ms": plan.estimated_recompute_overhead_ms,
                },
                "memory": {
                    "kind": "liveness_estimate_from_profiled_node_bytes",
                    "forward_peak_baseline_bytes": validation.forward_peak_before_bytes,
                    "forward_peak_checkpoint_bytes": validation.forward_peak_after_bytes,
                    "overall_peak_baseline_bytes": validation.overall_peak_before_bytes,
                    "overall_peak_checkpoint_bytes": validation.overall_peak_after_bytes,
                    "estimated_memory_saved_bytes": plan.estimated_memory_saved_bytes,
                },
                "latency": latency_report,
                "rewrite_status": rewrite_status,
                "plot_files": {
                    "forward_peak": os.path.abspath(save_path_fw),
                    "overall_peak": os.path.abspath(save_path_overall),
                    "memory_vs_opid": os.path.abspath(save_path_timeline),
                    "phase_memory": os.path.abspath(save_path_phase),
                    "components_baseline": os.path.abspath(save_path_components),
                    "components_checkpoint": os.path.abspath(save_path_components_cp),
                },
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(experiment_summary, f, indent=2)
            print(f"Saved experiment summary: {os.path.abspath(report_path)}")

        return gm

    def run(self):
        """Execute a single (non-compiled) training step for sanity checking."""
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


if __name__ == "__main__":
    # Usage: python benchmarks.py [model_name] [batch_size]
    # Defaults: Resnet152 with its standard batch size.
    name = sys.argv[1] if len(sys.argv) > 1 else model_names[1]
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else model_batch_sizes[name]
    print(f"Model: {name}, batch_size={bs}\n")
    exp = Experiment(name, bs)
    exp.init_opt_states()
    compiled_fn = compile(exp.train_step, exp.graph_transformation)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)
