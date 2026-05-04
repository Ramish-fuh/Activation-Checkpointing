# Handoff Prompt for New Agent

Copy this entire prompt and send it to the new agent in a fresh chat:

---

## Setup

You are continuing work on a **Harvard CS265 Big Data Systems activation checkpointing project**. 

**Key Context:**
- Project: PyTorch activation checkpointing with μ-TWO algorithm
- Current Phase: Phase 1 (profiling + diagnostics) — **Real training, not simulation**
- Status: Baseline profiling code ready; waiting for GPU validation
- Your role: Execute benchmarks on **real GPU training** and validate 5 core problems

**Important:** This project has a scope validation hook system that enforces Phase 1 boundaries. Read this first:
- Quick reference: [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) (2 min)
- Full details: [docs/HOOK-DESIGN.md](docs/HOOK-DESIGN.md)
- Testing: [docs/HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md)

---

## Source of Truth (Read These)

**Before you start, read these in order:**

1. **[docs/session_handoff.md](docs/session_handoff.md)** (5 min)
   - Current project state
   - What was done
   - Your exact next steps
   - The one decision you need to make

2. **[docs/MidWay_project_outline.md](docs/MidWay_project_outline.md)** (10 min)
   - Full Phase 1/2/3 scope definition
   - Technical approach
   - Expected deliverables

3. **[docs/notes.md](docs/notes.md)** (5 min)
   - Model specs
   - Batch sizes
   - Running context

---

## Phase 1 Validation Checklist

You are validating 5 core problems that Phase 1 solves. Run the baseline benchmark and check each one in sequence.

### Execute Real Training (Not Simulation)

```bash
# This runs ONE complete training step: forward + backward + optimizer
# On REAL GPU, with REAL data, producing REAL profiling metrics
python src/benchmarks.py Bert 4
```

This generates:
- `plots/classification_diagnostics_Bert_bs4.json` — Tensor classification and accounting
- `plots/memory_growth_Bert_bs4.png` — Tensor type growth over time
- `plots/memory_components_Bert_bs4.png` — Component breakdown + total-alive
- `plots/other_memory_components_Bert_bs4.png` — Top 20 OTHER tensors

---

## Problem 1: Graph IR & Observability ✓

**What we're solving:** Can we trace a complete training step (forward+backward+optimizer) into a single FX graph with clear region boundaries?

**How to validate:**
- Open `plots/classification_diagnostics_Bert_bs4.json`
- Check: `node_count` > 1000 (graph fully traced)
- Check: `sep_forward_index` and `sep_backward_index` exist (boundaries marked)
- Check: forward nodes < backward nodes (proper ordering)

**If valid:** ✅ Graph IR is correct. Move to Problem 2.  
**If invalid:** 🔧 Tracing failed. Check `src/graph_tracer.py` for SEPFunction markers.

---

## Problem 2: Operator Profiling ✓

**What we're solving:** Can we profile each node's runtime and output memory accurately?

**How to validate:**
- Open `plots/memory_vs_opid_Bert_bs4.png`
- Verify: Timeline shows gradual memory growth (no sudden spikes = profiling worked)
- Verify: Peaks labeled with `sep` and `sep_backward` markers
- Verify: All points in timeline have positive memory values

**Check JSON:**
```json
{
  "node_stats": [
    {
      "op_name": "...",
      "output_bytes": 12345678,  // Must be > 0
      "runtime_ms": 1.23
    }
  ]
}
```

**If valid:** ✅ Per-node profiling is accurate. Move to Problem 3.  
**If invalid:** 🔧 Profiling failed. Check `src/graph_prof.py` `run_node()` for CUDA event timing.

---

## Problem 3: Tensor Role Classification ✓

**What we're solving:** Can we reliably classify all tensors into 5 roles (PARAM, GRAD, ACT, OPT, OTHER)?

**How to validate:**
- Open `plots/classification_diagnostics_Bert_bs4.json`
- Check: `accounting_invariants.timeline_total_equals_sum_of_components == true`
  - This means: PARAM + GRAD + ACT + OPT + OTHER = TOTAL at every step
  - If false, classification is broken
- Check node breakdown:
  ```json
  {
    "node_classification": {
      "PARAM": N,      // Should be ~300-500 for BERT
      "GRAD": N,       // Should match PARAM count
      "ACT": N,        // Should be large (saved activations)
      "OPT": N,        // Should match PARAM count (optimizer state)
      "OTHER": N       // Should be small (temporaries)
    }
  }
  ```

**If accounting_invariants.timeline_total_equals_sum_of_components == true:** ✅ Classification is correct. Move to Problem 4.  
**If false:** 🔧 Classification bug. Fix `src/graph_prof.py` classification logic (Adam extraction, use-chain analysis).

---

## Problem 4: Activation Lifetime & Peak Memory ✓

**What we're solving:** Can we compute when each tensor is born/dies, and identify which types dominate at peak?

**How to validate:**
- Open `plots/memory_growth_Bert_bs4.png`
- Verify: You see 5 separate curves (PARAM, ACT, GRAD, OPT, OTHER) + TOTAL (black dashed)
- Verify: PARAM is flat (constant throughout)
- Verify: ACT rises during forward, falls during backward
- Verify: GRAD rises during backward, stays steady in optimizer
- Verify: Peak is labeled with step index

**Check JSON for peak breakdown:**
```json
{
  "forward_peak": {
    "step_index": 123,              // Which operation
    "total_mb": 520.8,              // Total peak
    "breakdown": {
      "PARAM": 44.6,
      "ACT": 331.2,                 // Usually dominant
      "GRAD": 44.6,
      "OPT": 0,
      "OTHER": 100.4
    }
  }
}
```

**If curves look reasonable + peak breakdown adds up:** ✅ Liveness analysis is correct. Move to Problem 5.  
**If not:** 🔧 Liveness bug. Check `src/graph_prof.py` `compute_peak_memory()` sweep algorithm.

---

## Problem 5: Experiment Orchestration & Reproducibility ✓

**What we're solving:** Can we run consistent benchmarks across models and batch sizes?

**How to validate:**
- Run SECOND benchmark with different model/batch:
  ```bash
  python src/benchmarks.py ResNet152 4
  ```
- Verify: New artifacts created (same filenames, different model name)
- Verify: Plots look reasonable (no crashes, OOM, or silent failures)
- Verify: JSON is valid and well-formed

**Check:** Can you reproduce the BERT run?
```bash
python src/benchmarks.py Bert 4  # Should produce same plots again
```
- Verify: New plots match previous ones (within ±5% variance due to GPU scheduling)

**If multiple models work + reproducible:** ✅ Benchmarking pipeline is solid. Move to OTHER validation.  
**If not:** 🔧 Orchestration bug. Check `src/benchmarks.py` `graph_transformation()` for inconsistent setup.

---

## The ONE Critical Decision: OTHER Memory

**What we're solving:** Is the `OTHER` memory spike legitimate, or is it a classification bug?

**How to validate:**
1. Open `plots/other_memory_components_Bert_bs4.png` (horizontal bar chart)
2. Identify top 3-5 tensors driving OTHER peak
3. For each tensor, ask: **Is this a legitimate temporary, or should it be ACT?**

**Legitimate OTHER tensors:**
- Backward pass temporaries (gradients computed mid-backward)
- Loss computation intermediates
- Optimizer fused Adam buffer states
- Placeholder/constant nodes from tracing

**Red flags (needs reclassification):**
- Top tensor has `"ACT"` in name but classified as OTHER
- Top tensor is consumed in backward but not classified as ACT
- OTHER peak is 2x larger than PARAM+GRAD+OPT combined

**Decision:**
- ✅ **Legitimate:** Document why and close Phase 1
- 🔧 **Needs fixing:** Update `src/graph_prof.py` classification, re-run benchmark, re-validate

**Report:**
- Screenshot of `other_memory_components_Bert_bs4.png`
- Top 3 OTHER tensors identified
- Your decision: legitimate or needs reclassification
- If reclassifying: run new benchmark and confirm fix

---

## If You Hit a Blocker

**Scenario 1: "I found a bug in Problem 3, but it requires Phase 2 logic"**
- Note: "Problem 3 classification broken, depends on checkpoint policy API"
- Document: What the bug is, why it depends on Phase 2
- Move on: Test Problems 4 and 5 with the broken classification as-is
- Reason: We validate each problem independently first, then fix cross-phase dependencies

**Scenario 2: "I can't validate X without running 100 benchmarks"**
- Note: "Need X benchmarks to fully validate, but can validate with Y for now"
- Document: What the limitation is
- Move on: Run your test and move to next problem
- Reason: Phase 1 is baseline validation, not exhaustive testing. Phase 2 does that.

**Scenario 3: "The GPU runs out of memory during profiling"**
- Note: "BERT batch 4 OOMs during profiling. Fallback to batch 2."
- Document: Batch size limit, where it fails
- Move on: Run with smaller batch, validate all 5 problems with reduced data
- Reason: We want baseline validation; batch size doesn't matter for architecture correctness

**Approach:** Validate what you can, document what you can't, move forward. Phase 1 is not exhaustive—it's foundational.

---

## File Editing Constraints

**What you CAN edit (Phase 1):**
- ✅ `src/benchmarks.py` — Benchmark harness improvements
- ✅ `src/graph_prof.py` — Classification logic fixes (Problem 3)
- ✅ `src/graph_tracer.py` — Tracing improvements (Problem 1)
- ✅ Create new plots or diagnostics
- ✅ Run benchmarks on different models/batch sizes (Problem 5)

**What you CANNOT edit (locked until Phase 1 closes):**
- ❌ `src/mu_two_policy.py` — Phase 2 work
- ❌ Graph rewriting logic — Phase 3 work
- ❌ `original/` folder — Read-only reference

**Why the lock?** The scope validation hook auto-enforces this. If you try to edit Phase 2/3 files, it asks you to discuss first.

---

## Structure: Sequential Validation

**Walk through in this order:**

1. ✓ Problem 1: Graph IR (tracing + boundaries)
2. ✓ Problem 2: Operator profiling (timing + memory per node)
3. ✓ Problem 3: Tensor classification (role accuracy)
4. ✓ Problem 4: Liveness & peak memory (temporal analysis)
5. ✓ Problem 5: Reproducibility (multiple models/batches)
6. ✓ **Critical decision:** OTHER memory legitimacy

**Each problem is independent.** If Problem 2 fails but Problem 1 passes, you know the issue is in profiling, not tracing.

---

## Real Training, Not Simulation

**Critical emphasis:** Every number you see is from REAL GPU execution, not synthetic data.

- ✅ Real forward pass: actual model, actual data, actual ops
- ✅ Real backward pass: actual gradient computation
- ✅ Real optimizer: actual parameter updates
- ✅ Real memory: actual GPU allocation / deallocation
- ✅ Real timing: CUDA event-based measurements

**Not synthetic:**
- ❌ No dummy tensors (all from model)
- ❌ No simulated gradients (computed via autograd)
- ❌ No approximated memory (measured with profiler)

This matters for publication quality: results are reproducible on any GPU with this codebase.

---

## Expected Outputs from Benchmark Run

When you run `python src/benchmarks.py Bert 4`, you get:

**Four PNG plots:**
- `memory_growth_Bert_bs4.png` — Shows 5 curves over time (Problem 4 proof)
- `memory_components_Bert_bs4.png` — Stacked bars + total (Problem 4 proof)
- `other_memory_components_Bert_bs4.png` — Top 20 OTHER tensors (Critical decision)
- `peak_memory_breakdown_fw_Bert_bs4.png` — Pie/bar at forward peak

**One JSON diagnostic file:**
- `classification_diagnostics_Bert_bs4.json` (1-2 MB)
  - Problems 1, 2, 3, 5 validation data
  - Critical: `accounting_invariants.timeline_total_equals_sum_of_components`

---

## Reference: The 5 Problems We're Solving

All problems are from [docs/MidWay_project_outline.md](docs/MidWay_project_outline.md) section 2:

| # | Problem | Phase 1 Validates |
|---|---|---|
| 1 | Graph IR & observability (torch.fx tracing) | Can we trace forward+backward+optimizer into one graph with clear boundaries? |
| 2 | Operator profiling (per-node compute/memory) | Do we have accurate timing and memory for each operation? |
| 3 | Tensor role classification (Adam semantics) | Are all tensors correctly classified into PARAM/GRAD/ACT/OPT/OTHER? |
| 4 | Activation lifetime & peak memory (liveness sweep) | Can we compute birth/death times and identify dominant types at peak? |
| 5 | Experiment orchestration (reproducible benchmarking) | Does the pipeline work across models and batch sizes? |

---

## Quick Reference

| Question | Answer |
|---|---|
| Full project scope? | [docs/MidWay_project_outline.md](docs/MidWay_project_outline.md) (section 2: Problems, section 3: Technical) |
| What exactly do I do right now? | [docs/session_handoff.md](docs/session_handoff.md) → "Handoff For GPU Agent" |
| How does the hook system work? | [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) (2 min read) |
| How do I test the hook system? | [docs/HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md) |
| Model specs and batch sizes? | [docs/notes.md](docs/notes.md) section 2: "Models Under Test" |
| What files can I edit? | This handoff, "File Editing Constraints" section |
| What if I hit a blocker? | This handoff, "If You Hit a Blocker" section |

---

## Hook System (Automatic, No Setup)

The codebase has a scope validation hook that:
- ✅ **PreToolUse:** Auto-blocks Phase 2/3 work during Phase 1 (if you try, it asks you to discuss first)
- ✅ **PostToolUse:** Auto-validates all outputs (syntax, JSON, artifacts)

**You don't need to do anything.** It runs automatically. You might see messages like:
- ✅ "Allow: Benchmark profiling approved"
- ✅ "Continue: All 4 plots created and valid"
- ⚠️ "Ask: Phase 2 work detected. Phase 1 only. Discuss?"

---

## Your Step-by-Step Workflow

### Phase 1: Preparation (15 min)
1. Read [docs/session_handoff.md](docs/session_handoff.md) → understand current state
2. Read [docs/MidWay_project_outline.md](docs/MidWay_project_outline.md) section 2 & 3 → understand the 5 problems
3. Skim [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) → know hook exists (don't worry about it yet)

### Phase 1: Execution (20 min)
4. Run: `python src/benchmarks.py Bert 4`
5. Wait for artifacts in `plots/`

### Phase 1: Validation (30 min)
6. Work through the **Phase 1 Validation Checklist** above in order:
   - Problem 1: Graph IR ✓
   - Problem 2: Operator profiling ✓
   - Problem 3: Classification ✓
   - Problem 4: Lifetime & peak ✓
   - Problem 5: Reproducibility ✓
   - **Critical:** OTHER memory legitimacy

7. For each problem: **If valid → move on. If invalid → fix and re-run.**

### Phase 1: Report (10 min)
8. Document your findings:
   - Which problems passed
   - Which need fixing (if any)
   - Screenshots of key plots
   - Your decision on OTHER legitimacy

---

## What "Passing" Looks Like

**All 5 problems pass:**
- ✅ Graph fully traced with boundaries
- ✅ Per-node profiling produces consistent numbers
- ✅ `accounting_invariants.timeline_total_equals_sum_of_components == true`
- ✅ 5 curves visible on memory_growth plot
- ✅ Multiple models/batches produce similar structure
- ✅ OTHER memory identified as legitimate

**Result:** Phase 1 is CLOSED. Phase 2 can begin.

---

## What Fixing Looks Like

**Problem 3 fails (accounting invariants false):**
1. Edit `src/graph_prof.py` classification logic
2. Re-run: `python src/benchmarks.py Bert 4`
3. Re-check accounting invariants
4. Repeat until true

**OTHER legitimacy uncertain:**
1. Document which tensors are questionable
2. Cross-reference with `src/graph_prof.py` logic
3. Update classification if needed
4. Re-validate

**Problem 5 fails (resnet doesn't work):**
1. Run: `python src/benchmarks.py ResNet152 2` (smaller batch)
2. If works: document batch limit and move on
3. If still fails: trace issue in `src/benchmarks.py`

**Philosophy:** Fix only what you need to validate the problem. Document any limitations and move forward.

---

## When to Ask for Help

- Something crashes: **Describe the error and which problem you were on**
- Accounting invariants won't be true: **Show your classification logic fix attempt**
- OTHER decision uncertain: **Describe the top 3 tensors and their usage patterns**
- GPU OOMs: **Show batch size limit and fallback configuration**
- Not sure if something is Phase 1: **The hook will tell you if you try it**

---

## Summary

You are validating a research-grade profiling pipeline on real GPU training. You'll work through **5 concrete problems**, each with a **clear validation checklist**. If something isn't validated yet due to Phase 2 dependencies, you **note it and move on**—don't block on it.

**Execution path:**
1. One benchmark run (real training, not simulation)
2. Five validation checklists (Problems 1-5)
3. One critical decision (OTHER legitimacy)
4. Report findings

**If all pass:** Phase 1 closes, Phase 2 begins.  
**If something fails:** Fix it, re-run, re-validate.  
**If you hit a blocker:** Document it and move forward.

**Time estimate:** 1.5 hours total (prep + run + validation + documentation)

---

**Previous commits:**
- `44156ff` — Agent handoff prompt
- `4c97e6c` — Hook system deployed
- `981f898` — Data-driven validation added
- `e72ba05` — Previous work

**Branch:** `cuda2`  
**Status:** Phase 1 ready for GPU validation  
**Next:** Run benchmark, validate 5 problems, make OTHER decision

**Ready to go!**
