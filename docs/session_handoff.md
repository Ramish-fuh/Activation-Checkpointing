# Session Handoff - Activation Checkpointing Project

## Role And Working Style

- The student is working on a Harvard CS265 / Big Data Systems activation
  checkpointing project.
- Act like a professor/mentor: take charge when the evidence is confusing,
  explain reasoning clearly, and help the student understand each phase.
- Do not rush into policy or graph rewriting before the baseline memory
  accounting is explainable.
- Important project constraint: do not modify the `original/` folder. Treat it
  as the starter-code reference.
- It is acceptable to edit starter code under `src/` when needed.
- Avoid creating lots of new files. The current extra documentation files are
  intentional.

## Current Decision

- We restarted from the original starter code and are stopping at **Phase 1
  only** for now.
- Phase 1 means:
  - trace one full training step;
  - classify tensors;
  - profile node runtime and output memory;
  - compute liveness and peak memory;
  - generate diagnostic plots;
  - explain `OTHER`.
- Do **not** work on mu-TWO policy, checkpoint selection, or FX graph rewrite
  until Phase 1 plots and diagnostics are correct.

## Handoff For GPU Agent

- The active branch is `cuda2`.
- The benchmark path now profiles a real traced training step, not a synthetic
  simulation.
- `src/benchmarks.py` runs the compiled forward + backward + optimizer step,
  profiles it with `GraphProfiler`, and writes Phase 1 plots plus JSON
  diagnostics.
- The new emphasis is not on checkpointing yet; it is on proving the baseline
  accounting and explaining why `OTHER` spikes.
- The current hypothesis is that some tensors being classified as `OTHER` are
  actually the source of the peak, or are at least masking the real growth
  pattern across tensor types.

What to inspect first on GPU:

- Run `python src/benchmarks.py Bert 4`.
- Open `plots/classification_diagnostics_Bert_bs4.json`.
- Open `plots/memory_growth_Bert_bs4.png` to see how PARAM, ACT, GRAD, OPT,
  and OTHER grow over time.
- Open `plots/other_memory_components_Bert_bs4.png` to identify the largest
  tensors counted as `OTHER` at the forward peak.
- Cross-check `plots/memory_components_Bert_bs4.png` against the new total line
  so component growth is compared with whole-graph growth.

Goal for the next pass:

- Determine exactly what the top `OTHER` tensors are.
- Decide whether they are legitimate placeholders / temporaries / backward
  temps, or whether some should be reclassified as ACT.
- If the spike is due to misclassification, fix the profiler classification
  before moving on to any checkpoint policy work.

## Why We Restarted

- Previous plots had become hard to defend:
  - `Peak Memory Breakdown` and `Memory Components vs Operations` were being
    compared incorrectly.
  - The breakdown reports the sum of all live components at one peak operation.
  - The component plot shows each component's separate curve over time.
  - Without a total-memory curve on the component plot, it was easy to mistake a
    component peak for the total peak.
- `OTHER` was very large and needed decomposition before we could claim it was
  legitimate.
- Checkpointing activations should not magically remove unrelated `OTHER`
  tensors. The baseline accounting must prove what is live and why before
  checkpointing is modeled.

## Active Files And Current State

- `src/benchmarks.py`
  - Rebuilt as a Phase 1 baseline-only benchmark harness.
  - Supports `Bert` and `Resnet152`.
  - Does not import or run mu-TWO policy.
  - Does not apply activation checkpoint graph rewriting.
  - Writes plots and JSON diagnostics to `plots/`.
  - Adds a type-growth plot so PARAM / ACT / GRAD / OPT / OTHER can be tracked
    together over time.

- `src/graph_prof.py`
  - Rebuilt as the Phase 1 profiler.
  - Detects `separator.sep` and `separator.sep_backward`.
  - Extracts PARAM, GRAD, and OPT roles from fused Adam arguments.
  - Classifies ACT as forward/loss tensor storage consumed by backward.
  - Follows alias/view/getitem-style nodes back to their storage owner.
  - Computes liveness ranges from profiled output bytes.
  - Adds exact peak operation/node/target reporting.
  - Adds a `total alive` line to memory component plots.
  - Adds `OTHER` memory decomposition over operations.
  - Emits classification diagnostics JSON.

- `src/graph_tracer.py`
  - Restored close to original starter structure.
  - Preserves erased `dummy.tag_grad` metadata so the gradient accumulation
    diagnostic can show parameter-gradient production.

- `src/mu_two_policy.py`
  - Still present in the tree for later policy work, but no longer part of the
    active benchmark path on CUDA2.
  - Reason: keep the next policy implementation separate until Phase 1 plots
    and accounting are fully validated.

- `docs/checkpointing_findings.md`
  - Reset to current active findings only.
  - Documents the Phase 1 restart, what changed, and what to inspect next.

## Verification Already Done

- Syntax compilation passed with the project venv:

```bash
.venv/bin/python -m py_compile src/graph_prof.py src/benchmarks.py src/graph_tracer.py src/activation_checkpoint.py src/utils.py
```

- Imports passed in the project venv:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from graph_prof import GraphProfiler; print('graph_prof import ok')"
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import graph_tracer; print('graph_tracer import ok')"
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import benchmarks; print('benchmarks import ok')"
```

- The system `python3` did not have `torch`; use the venv or the Kaggle/project
  environment.

## Next Run

Run BERT first:

```bash
python src/benchmarks.py Bert 4
```

ResNet-152 should wait until BERT's accounting is explainable.

## Artifacts To Inspect First

After running BERT batch size 4, inspect these in order:

- `plots/classification_diagnostics_Bert_bs4.json`
- `plots/memory_growth_Bert_bs4.png`
- `plots/memory_vs_opid_Bert_bs4.png`
- `plots/memory_components_Bert_bs4.png`
- `plots/peak_memory_breakdown_fw_Bert_bs4.png`
- `plots/other_memory_components_Bert_bs4.png`
- `plots/gradient_accumulation_Bert_bs4.png`

## What Must Be True Before Phase 2

- In `classification_diagnostics_Bert_bs4.json`:
  - `accounting_invariants.timeline_total_equals_sum_of_components` must be
    `true`.
  - `forward_peak.total_bytes` must equal the sum of its role breakdown.
  - The `forward_peak.step_index` must match the total-memory peak on
    `memory_vs_opid_Bert_bs4.png`.
  - `forward_peak.top_other_alive` must explain the largest `OTHER` tensors.

- In `memory_components_Bert_bs4.png`:
  - The black `total alive` line should match the peak breakdown.
  - Individual component peaks should not be mistaken for total-memory peaks.

- In `other_memory_components_Bert_bs4.png`:
  - Check whether `OTHER` is mostly logits/loss tensors, placeholders,
    non-saved forward temporaries, optimizer temps, or backward temps.
  - If top `OTHER` tensors have backward consumers and should be saved
    activations, fix classification before moving on.

## Key Interpretation To Keep

- `PARAM`, `GRAD`, and `OPT` come from fused Adam semantics.
- `ACT` means tensor storage created in forward/loss and later consumed by
  backward.
- `OTHER` does not automatically mean wrong. It can include:
  - input and label placeholders;
  - logits or loss temporaries not saved for backward;
  - non-saved forward temporaries;
  - optimizer/update temporaries;
  - backward temporary tensors;
  - alias/view metadata with zero storage.
- A high `OTHER` value is only a bug if diagnostics show that checkpointable
  activations are being left in `OTHER`.

## Current Git Working Tree Context

Expected working tree changes on the current CUDA2 handoff:

- Modified:
  - `src/benchmarks.py`
  - `src/graph_prof.py`
  - `docs/session_handoff.md`
- Added:
  - `docs/checkpointing_findings.md`

This is intentional. The handoff stays focused on Phase 1 accounting and the
growth of each tensor type before any policy or rewrite work is resumed.
