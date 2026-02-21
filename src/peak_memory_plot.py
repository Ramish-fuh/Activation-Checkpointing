"""
Phase 1 Experimental Analysis:
  - Profile DummyModel, Resnet18, and Transformer at multiple batch sizes
  - Print profiling stats (4a deliverable)
  - Generate peak memory breakdown vs mini-batch size charts (4b deliverable)
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.fx as fx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from graph_prof import GraphProfiler, NodeType
from graph_tracer import SEPFunction, compile
from utils import get_device, get_optimizer_kwargs


# ---------------------------------------------------------------------------
# Model setup helpers (mirror benchmarks.py but avoid import to keep it clean)
# ---------------------------------------------------------------------------

class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


def make_experiment(model_name: str, batch_size: int):
    """Return (model, example_inputs, train_step_fn, optimizer) for a model."""
    dev = get_device()

    if model_name == "DummyModel":
        dim, layers = 100, 10
        model = DummyModel(dim=dim, layers=layers).to(str(dev))
        batch = torch.randn(batch_size, dim, device=dev)
        example_inputs = (batch,)

        def train_step(model, opt, inputs):
            loss = model(inputs[0]).sum()
            loss = SEPFunction.apply(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    elif model_name == "Resnet18":
        from torchvision.models import resnet18
        num_classes = 10
        inp = torch.randn(batch_size, 3, 224, 224, device=dev)
        target = torch.randint(0, num_classes, (batch_size,), device=dev)
        with torch.device(dev):
            model = resnet18()
        example_inputs = (inp, target)

        def train_step(model, opt, inputs):
            loss = F.cross_entropy(
                model(inputs[0]),
                inputs[1],
                ignore_index=-1,
            )
            loss = SEPFunction.apply(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    elif model_name == "Transformer":
        from torch.testing._internal.distributed._tensor.common_dtensor import (
            ModelArgs, Transformer,
        )
        vocab_size = 2048
        seq_len = 256
        with torch.device(dev):
            model_args = ModelArgs(
                n_layers=8, n_heads=4, vocab_size=vocab_size,
                max_seq_len=seq_len, dropout_p=0.1,
            )
            model = Transformer(model_args)
        src = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        example_inputs = (src, tgt)

        def train_step(model, opt, inputs):
            logits = model(inputs[0])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs[1].view(-1),
                ignore_index=-1,
            )
            loss = SEPFunction.apply(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = optim.Adam(model.parameters(), lr=1e-2, **get_optimizer_kwargs())

    # Initialize optimizer states
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.rand_like(param)
    optimizer.step()
    optimizer.zero_grad()

    return model, example_inputs, train_step, optimizer


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_model(
    model_name: str,
    batch_size: int,
    print_full_stats: bool = False,
) -> Tuple[int, Dict[NodeType, int], str]:
    """Profile a model at a given batch size.

    Returns (peak_bytes, breakdown_dict, stats_text).
    """
    torch.manual_seed(20)
    model, inputs, train_step, optimizer = make_experiment(model_name, batch_size)

    profiler_holder: Dict[str, GraphProfiler] = {}

    def graph_transformation(gm: fx.GraphModule, args: Any) -> fx.GraphModule:
        gp = GraphProfiler(gm)
        warm_up, measure = 2, 3
        with torch.no_grad():
            for _ in range(warm_up):
                gp.run(*args)
            gp.reset_stats()
            for _ in range(measure):
                gp.run(*args)
            gp.aggregate_stats()
        profiler_holder["gp"] = gp
        return gm

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optimizer, inputs)

    gp = profiler_holder["gp"]

    # Capture print_stats output to a string
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    gp.print_stats()
    sys.stdout = old_stdout
    stats_text = buf.getvalue()

    if print_full_stats:
        print(stats_text)

    peak_mem, breakdown = gp.compute_peak_memory()
    return peak_mem, breakdown, stats_text


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_peak_memory(
    model_name: str,
    batch_sizes: List[int],
    results: List[Tuple[int, Dict[NodeType, int]]],
    out_path: str,
):
    param_mb = [r[1].get(NodeType.PARAM, 0) / 1024**2 for r in results]
    act_mb   = [r[1].get(NodeType.ACT, 0)   / 1024**2 for r in results]
    grad_mb  = [r[1].get(NodeType.GRAD, 0)  / 1024**2 for r in results]
    other_mb = [r[1].get(NodeType.OTHER, 0)  / 1024**2 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(batch_sizes))

    ax.bar(x, param_mb, label="Parameters", color="#2196F3")
    ax.bar(x, act_mb, bottom=param_mb, label="Activations", color="#FF9800")
    bottom2 = [p + a for p, a in zip(param_mb, act_mb)]
    ax.bar(x, grad_mb, bottom=bottom2, label="Gradients", color="#4CAF50")
    bottom3 = [b + g for b, g in zip(bottom2, grad_mb)]
    ax.bar(x, other_mb, bottom=bottom3, label="Other", color="#9E9E9E")

    ax.set_xlabel("Mini-batch Size")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"Peak Memory Breakdown — {model_name} (No AC)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    docs = "../docs"

    experiments = {
        "DummyModel":  [100, 250, 500, 750, 1000],
        "Resnet18":    [1, 2, 4, 8, 16],
        "Transformer": [1, 2, 4],
    }

    for model_name, batch_sizes in experiments.items():
        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        # 1. Full profiling stats at default batch size (first in list for real models)
        default_bs = batch_sizes[len(batch_sizes) // 2]
        print(f"\n  Full profiling stats at batch_size={default_bs}:")
        _, _, stats_text = profile_model(model_name, default_bs, print_full_stats=True)

        # Save stats to file
        stats_path = f"{docs}/profiling_stats_{model_name}.txt"
        with open(stats_path, "w") as f:
            f.write(f"Model: {model_name}, batch_size={default_bs}\n")
            f.write(stats_text)
        print(f"  Stats saved: {stats_path}")

        # 2. Peak memory at multiple batch sizes
        print(f"\n  Peak memory sweep: batch_sizes={batch_sizes}")
        results = []
        for bs in batch_sizes:
            print(f"    batch_size={bs} ...", end=" ", flush=True)
            peak, breakdown, _ = profile_model(model_name, bs)
            results.append((peak, breakdown))
            print(f"peak = {peak / 1024**2:.2f} MB")

        # 3. Generate chart
        chart_path = f"{docs}/peak_memory_{model_name}.png"
        plot_peak_memory(model_name, batch_sizes, results, chart_path)

    print(f"\nDone. All outputs saved under {docs}/")


if __name__ == "__main__":
    main()
