**Project:** Activation Checkpointing for DNN Training (CS265)

*Phase 1*

## End-to-End Flow (How It Is Done + Functions Used)

1. **Build and trace one full training-step graph.**
   - How: `make_fx` traces one complete training step (forward + loss + backward + optimizer) into a single FX graph. During tracing, `SEPFunction.apply()` inserts two sentinel ops — `separator.sep` at the end of forward and `separator.sep_backward` at the start of backward — so region boundaries can be found later. The same graph is then profiled node-by-node.
   - Functions: `compile()` / `_compile()` in `src/graph_tracer.py`; `graph_transformation()` in `src/benchmarks.py`; `GraphProfiler.run()` / `run_node()` in `src/graph_prof.py`.
   - Why: one unified graph exposes cross-boundary dependencies (forward tensors consumed in backward) that are invisible when forward and backward run separately.

2. **Initialize graph analysis structures.**
   - How: convert graph nodes to an ordered list and index map. **No sorting is performed** — `fx.Graph.nodes` is already in topological order because `make_fx` records operations in the exact execution order they ran during tracing. An operation can only run after its inputs exist, so the recorded order is topological by construction.
   - Functions: `GraphProfiler.__init__()` calling `_build_node_index()` in `src/graph_prof.py`.
   - Why: `node_index[node]` gives O(1) position lookup that every later pass uses for positional comparisons: is a consumer before or after `sep_backward`? what is a tensor's liveness window? where is the peak memory step?

3. **Detect graph regions (forward, loss, backward).**
   - How: locate `separator.sep` and `separator.sep_backward`, then label each node by region.
   - Functions: `_find_region_boundaries()`, `_label_regions()` in `src/graph_prof.py`.
   - Why: boundary detection is the base for correct activation and lifetime logic.

4. **Detect PARAM and GRAD nodes.**
   - How: read `_fused_adam` argument lists and collect param, grad, and optimizer-state nodes.
   - Functions: `_identify_params_and_grads()`, `_try_fused_adam()`, `_collect_fx_nodes()` in `src/graph_prof.py`.
   - Why: placeholder type alone is ambiguous, optimizer semantics are the reliable source. 
  
   - `_fused_adam` serves a dual role here: at runtime it is the actual weight update (one fused CUDA kernel); statically it is the only node in the graph that explicitly groups tensors by role (`args[0]` = params, `args[1]` = grads, `args[2–4]` = moment estimates), so we piggyback on its argument contract to bootstrap all downstream classification.

5. **Run node classification (including ACT nodes).**
   - How: classify nodes into PARAM, ACT, GRAD, OTHER; for ACT record `last_fw_access` and `first_bw_access`.
   - Functions: `_classify_nodes()`, `_is_intermediate_activation()`, `_register_activation()` in `src/graph_prof.py`.
   - Why: activation checkpointing depends on identifying tensors that cross forward to backward.

6. **Collect runtime compute and memory stats node-by-node.**
   - How: during execution, each node is timed with CUDA events and output memory is measured in bytes.
   - Functions: `run_node()`, `_create_cuda_events()`, `_tensor_bytes()` in `src/graph_prof.py`.
   - Why: this provides operator-level evidence for compute vs memory tradeoffs.

7. **Aggregate and print profiling statistics.**
   - How: after warmup, average per-node runtimes and print structured profiling summaries.
   - Functions: `reset_stats()`, `aggregate_stats()`, `print_stats()` in `src/graph_prof.py`; called from `graph_transformation()` in `src/benchmarks.py`.
   - Why: midpoint check-in requires clear profiling and static-analysis outputs.

8. **Compute peak memory and generate artifacts.**
   - How: simulate tensor liveness across graph steps and compute peak-by-type breakdown, then save a plot.
   - Functions: `compute_peak_memory()`, `_build_alive_ranges()`, `_sweep_for_peak()`, `plot_peak_memory_breakdown()` in `src/graph_prof.py`; output path handled in `src/benchmarks.py`.
   - Why: this is the concrete deliverable for peak memory analysis.

9. **Why we made it modular.**
   - How: each stage is isolated (trace, static analysis, classification, runtime stats, aggregation, memory model, plotting).
   - Modules: `src/graph_tracer.py` for graph construction, `src/graph_prof.py` for analysis/profiling, `src/benchmarks.py` for experiment orchestration.
   - Why modular: clearer flow, easier debugging per stage, safer edits, and direct reuse for Phase 2/3 (checkpoint policy + graph rewrite).



