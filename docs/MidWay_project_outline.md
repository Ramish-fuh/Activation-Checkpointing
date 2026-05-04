\begin{center}
	{\LARGE Milestone 1}\\
	{CSCI E-265, Spring 2026}\\
	{Ramish Fuh}
\end{center}

## 1. Introduction
This milestone focuses on Phase 1 of the CS265 ML systems project: building a robust profiling and analysis pipeline for activation checkpointing in PyTorch training graphs. The objective is to trace one full training step (forward, loss, backward, and optimizer), collect operator-level compute and memory statistics, classify tensors by role (parameters, gradients, activations, and other), and compute peak memory behavior across graph execution. This profiler is the foundation for the later phases, where we will run the mu-TWO policy to decide retained activations and rewrite the graph to insert recomputation subgraphs.

## 2. Problems Tackled
- [Graph intermediate representation (IR) and observability design] We need a PyTorch-native intermediate representation (IR) that is easy to trace and rewrite, plus explicit forward/backward boundary visibility in one graph so cross-boundary dependencies are analyzable.
- [Operator profiling] We need accurate per-node runtime and output memory measurements to reason about memory-compute tradeoffs.
- [Role classification via Adam semantics] We need reliable tensor-role labels, so we use Adam update signatures to extract params, grads, and optimizer state directly, then combine that with graph-region/use-chain analysis to classify nodes into PARAM, GRAD, ACT, OPT_STATE, or OTHER.
- [Activation lifetime and peak-memory analysis] We need first/last-use liveness information and a repeatable sweep method to identify where memory peaks and which tensor classes dominate the peak.
- [Experiment orchestration and reproducibility] We need a consistent benchmarking pipeline across models and batch sizes so outputs are comparable and report-ready.

## 3. Technical Description

### Problem 1: Graph intermediate representation (IR) design decision and unified training-step construction
- a) Problem framing: We first had to choose the graph representation. Activation checkpointing decisions depend on cross-boundary use chains where tensors produced during forward are consumed during backward. If forward and backward are profiled separately, or if we use an IR that is hard to transform back into PyTorch execution, activation ownership/lifetimes cannot be computed or rewritten correctly.
- b) High-level solution: We explicitly chose `torch.fx` (`make_fx`) as the IR because it is native to PyTorch, keeps operator-level nodes, and supports later graph rewriting. We then trace one complete training step and insert explicit markers to delimit regions: forward end and backward start. This gives a dependency-valid linearized order for boundary-aware analysis, even when some model operators can execute with runtime overlap.
- c) Deeper details:
  - We selected `torch.fx` because it is directly integrated with PyTorch execution, gives node-level introspection without exporting to another framework, and supports later graph rewriting for checkpointing.
  - We insert sentinel ops (`sep` and `sep_backward`) during tracing to identify region boundaries deterministically.
  - The resulting FX node order is topological for data dependencies, so we use index-based comparisons instead of expensive graph traversals for many analyses; however, it does not directly encode hardware-level parallel scheduling.
  - This representation supports both profiling (Phase 1) and transformation (Phase 3) in the same intermediate format.
  - Tradeoff: Choosing `torch.fx` gives strong transformability and observability, but ties us to PyTorch internals and version-specific trace/operator behavior that can change across environments.

### Problem 2: Per-operator compute and memory profiling
- a) Problem framing: The checkpointing policy needs cost signals. Without operator runtime and memory output statistics, we cannot choose between retaining and recomputing activations in a principled way.
- b) High-level solution: Execute the traced graph through an interpreter that runs each node individually and records runtime and output tensor bytes per node, then aggregate over warmup and measured iterations.
- c) Deeper details:
  - Runtime is captured with CUDA events around each node execution to reduce host-side timing noise.
  - Memory is measured as output tensor bytes (`numel * element_size`) and accumulated by node/category.
  - We report per-node and aggregate summaries so we can diagnose outliers (expensive ops, large activation producers) and produce midpoint evidence.
  - Tradeoff: Node-level profiling improves policy quality, but adds runtime overhead during profiling runs and can under-represent true kernel-overlap effects in highly parallel models.

### Problem 3: Role classification using Adam semantics
- a) Problem framing: Placeholder nodes alone are ambiguous in FX traces, but checkpointing needs exact tensor roles for memory accounting and policy constraints. We therefore need one robust source of truth for role inference.
- b) High-level solution: We use Adam-family optimizer call semantics as that source of truth, because optimizer arguments naturally encode parameter, gradient, and optimizer-state groups. We then complete classification with graph-region and use-chain analysis.
- c) Deeper details:
  - In Adam update calls, argument groups expose roles directly (parameters, gradients, and moment/state tensors), so extraction is deterministic rather than heuristic.
  - We map those extracted groups to PARAM, GRAD, and OPT_STATE sets, then classify forward intermediates consumed in backward as ACT.
  - Any remaining nodes are labeled OTHER so every node belongs to exactly one role class.
  - For ACT nodes, we record `last_fw_access` and `first_bw_access`, which supports liveness intervals and later recomputation placement.
  - Tradeoff: Adam semantics make role extraction robust and simple, but this approach is less portable to optimizers whose update signatures do not expose role groupings as clearly.

### Problem 4: Activation lifetime analysis and peak memory modeling
- a) Problem framing: Runtime snapshots alone do not explain why peak happens or which tensor classes dominate at peak. We need explicit activation lifetimes (first/last use) and an explainable liveness-based model to reason about peak memory.
- b) High-level solution: Record first/last-use lifetime metadata, build alive intervals from node production/consumption positions, then sweep execution order and accumulate bytes of all live tensors by category at each step.
- c) Deeper details:
  - Intervals are generated using node indices and region-aware first/last-use metadata.
  - The sweep algorithm outputs both global peak value and per-category contribution at the peak step.
  - This enables required midpoint visualizations (peak memory breakdown and batch-size trend without activation checkpointing).
  - Tradeoff: Lifetime-based peak modeling is explainable and fast, but it is an approximation of allocator-level GPU behavior and may differ from absolute device-reported peak memory.

### Problem 5: Experiment orchestration and reproducibility
- a) Problem framing: Profiling outputs are only useful if generated consistently across models and batch sizes with repeatable settings.
- b) High-level solution: Centralize experiment setup and reporting in benchmark drivers that call tracing, profiling, aggregation, and plotting in a fixed order.
- c) Deeper details:
  - We apply the same measurement pipeline to DummyModel, ResNet variants, and Transformer workloads.
  - Outputs are saved as stats reports and figures for direct use in midway and final documentation.
  - The modular split (tracer, profiler, benchmark runner) minimizes coupling and prepares clean extension points for Phase 2/3.
  - Tradeoff: A unified pipeline improves reproducibility and maintainability, but enforces shared assumptions that may need model-specific overrides for edge cases or future optimizer variants.

## 4. Challenges
- Optimizer path variability (`foreach` vs `fused`) can change graph signatures and break assumptions in param/grad extraction unless handled defensively.
- PyTorch graph structure can differ by version, CUDA backend, and model shape, requiring robust pattern checks rather than brittle op-name matching.
- GPU memory behavior includes allocator effects and asynchronous execution; measured node output bytes and true device peak are related but not identical.
- Larger models and batch sizes can trigger OOM before full profiling completes, so experiments need staged scaling and fallback settings.
- Recomputation planning in later phases must preserve numerical correctness and avoid invalid rewrites around side effects or in-place operations.

## 5. Midway Deliverable Mapping
- Completed: profiling statistics for compute/memory per operator.
- Completed: tensor classification and activation lifetime static analysis.
- Completed: peak-memory-by-type analysis pipeline and plotting support.
- peak memory vs mini-batch size bar graph (without activation checkpointing) across selected model.

## Graph

![Peak memory breakdown for ResNet-18, batch size 16](docs/peak_memory_breakdown_Resnet18_bs16.png)

\newpage

## Stats

### Memory Summary

| Category | Value |
| --- | --- |
| PARAM | 44.6 MB |
| ACT | 331.2 MB |
| GRAD | 44.6 MB |
| OTHER | 100.4 MB |
| PEAK | 520.8 MB |

### Graph Profiling Results

#### Node Classification

| Class | Count |
| --- | --- |
| PARAM | 62 nodes |
| ACT | 104 nodes |
| GRAD | 62 nodes |
| OTHER | 947 nodes |

For full stats, see attachment.

## 6. Next-Step Bridge to Phase 2 and 3
- Phase 2: consume profiler signals in a mu-TWO-style retention/recomputation policy.
- Phase 3: extract minimal recomputation subgraphs and insert them before first backward use of discarded activations.
- Validation: compare memory reduction and iteration-latency overhead with and without activation checkpointing across batch sizes.

