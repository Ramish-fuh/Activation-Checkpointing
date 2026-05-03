# Handoff Prompt for New Agent

Copy this entire prompt and send it to the new agent in a fresh chat:

---

## Setup

You are continuing work on a **Harvard CS265 Big Data Systems activation checkpointing project**. 

**Key Context:**
- Project: PyTorch activation checkpointing with μ-TWO algorithm
- Current Phase: Phase 1 (profiling + diagnostics)
- Status: Baseline profiling is ready; waiting for GPU validation
- Your role: Execute benchmarks on GPU and validate the OTHER memory classification

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

## What to Do Next (Immediate)

### Your Job
You have a GPU. The previous team doesn't. Your task:

1. **Run the baseline benchmark:**
   ```bash
   python src/benchmarks.py Bert 4
   ```

2. **Inspect 4 key artifacts:**
   - `plots/classification_diagnostics_Bert_bs4.json` — Check if accounting_invariants.timeline_total_equals_sum_of_components == true
   - `plots/memory_growth_Bert_bs4.png` — See how PARAM/ACT/GRAD/OPT/OTHER grow over time
   - `plots/memory_components_Bert_bs4.png` — Verify total-alive line (dashed black) matches sum of parts
   - `plots/other_memory_components_Bert_bs4.png` — Identify top 3-5 OTHER tensors at forward peak

3. **Make the diagnostic decision:**
   Are the top OTHER tensors legitimate (backward temps, placeholders, optimizer temporaries)?
   - YES → Document and close Phase 1
   - NO → We need to fix tensor classification in `src/graph_prof.py`

4. **Report back:**
   - Screenshots of the 4 plots
   - JSON accounting_invariants result
   - Your decision on OTHER legitimacy
   - Any blockers or surprises

---

## Hook System (Don't Worry About It, It's Automatic)

The codebase has a scope validation hook that:
- ✅ **PreToolUse:** Blocks Phase 2/3 work until Phase 1 closes (you can't accidentally start checkpoint policy yet)
- ✅ **PostToolUse:** Validates all outputs (syntax, JSON, artifacts)

**You don't need to do anything.** It runs automatically. If you try something out of scope, it'll ask you to discuss first.

---

## Key Files (You'll Edit These)

| File | Phase | Role |
|---|---|---|
| `src/benchmarks.py` | 1 | Benchmark harness (profile + plot) |
| `src/graph_prof.py` | 1 | Profiler + classification logic |
| `src/graph_tracer.py` | 1 | Trace forward+backward+optimizer |
| `src/mu_two_policy.py` | 2 | **LOCKED** until Phase 1 closes |
| `plots/` | 1 | Output artifacts (read these, don't edit) |

---

## Expected First Run Results

When you run `python src/benchmarks.py Bert 4`, you should see:

✅ **New artifacts created:**
- `plots/classification_diagnostics_Bert_bs4.json` — 1-2 MB, detailed diagnostics
- `plots/memory_growth_Bert_bs4.png` — Show 5 curves (PARAM, ACT, GRAD, OPT, OTHER, TOTAL)
- `plots/memory_components_Bert_bs4.png` — Stacked bars + total-alive dashed line
- `plots/other_memory_components_Bert_bs4.png` — Horizontal bar chart of top 20 OTHER tensors

✅ **JSON structure:**
```json
{
  "accounting_invariants": {
    "timeline_total_equals_sum_of_components": true,  // This must be true
    "peak_index_matches_visual": true
  },
  "node_classification": {
    "PARAM": 1234,
    "ACT": 567,
    "GRAD": 1234,
    "OPT": 5678,
    "OTHER": 89
  },
  "forward_peak": {
    "step_index": 123,
    "total_mb": 520.8,
    "top_other_alive": [
      {"name": "tensor_name", "mb": 100.4, "reason": "..."}
    ]
  }
}
```

---

## The ONE Decision You Need to Make

**Question:** Is the `OTHER` memory spike legitimate?

**Legitimate reasons (OK):**
- Backward pass temporaries (autograd internals)
- Placeholder nodes from tracing artifacts
- Optimizer intermediate states (fused Adam buffers)
- Loss computation intermediates

**Red flags (needs fixing):**
- Top OTHER tensors have backward consumers but aren't classified as ACT
- OTHER peak is much larger than PARAM+GRAD+OPT combined (suggests classification bug)
- Top OTHER tensor list shows activation-like names

**How to decide:**
1. Look at `plots/other_memory_components_Bert_bs4.png`
2. Identify top 3-5 tensors driving the OTHER peak
3. Cross-reference with `src/graph_prof.py` classification logic
4. Make call: legit or needs reclassification

---

## Constraints (Always Valid)

- ❌ **DO NOT** edit `src/mu_two_policy.py` (Phase 2, locked during Phase 1)
- ❌ **DO NOT** start graph rewriting work (Phase 3)
- ❌ **DO NOT** modify `original/` folder (read-only reference)
- ✅ **DO** edit `src/benchmarks.py`, `src/graph_prof.py`, `src/graph_tracer.py` if needed
- ✅ **DO** create new plots or diagnostics
- ✅ **DO** run benchmarks on different models/batch sizes

---

## Questions?

| Question | Answer |
|---|---|
| What's the full project scope? | [docs/MidWay_project_outline.md](docs/MidWay_project_outline.md) |
| What exactly do I do right now? | [docs/session_handoff.md](docs/session_handoff.md) → "Handoff For GPU Agent" section |
| How does the hook system work? | [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) |
| How do I test the hook system? | [docs/HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md) |
| What models are we using? | [docs/notes.md](docs/notes.md) → "Models Under Test" section |

---

## Ready to Start?

1. Read [docs/session_handoff.md](docs/session_handoff.md) (5 min)
2. Run `python src/benchmarks.py Bert 4` (5-10 min)
3. Inspect the 4 plots + JSON
4. Tell me your decision on OTHER legitimacy
5. We close Phase 1 and move to Phase 2

**Let's go!**

---

**Previous commits:**
- `4c97e6c` — Hook system deployed
- `981f898` — Data-driven validation added
- `e72ba05` — Previous work

**Branch:** `cuda2`  
**Checkpoint:** Phase 1 baseline ready for GPU validation
