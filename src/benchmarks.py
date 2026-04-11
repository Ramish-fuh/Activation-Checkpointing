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
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
from torchvision.models import resnet152
from graph_prof import GraphProfiler
from mu_two_policy import build_checkpoint_plan, PolicyConfig
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
            plan = build_checkpoint_plan(graph_profiler, PolicyConfig())

            plots_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'plots'
            )
            os.makedirs(plots_dir, exist_ok=True)

            plan_path = os.path.join(
                plots_dir,
                f"checkpoint_plan_{self.model_name}_bs{self.batch_size}.json",
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, indent=2)
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
