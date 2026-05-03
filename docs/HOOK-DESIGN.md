# CS265 Project Scope Validation Hook

> **Quick Navigation:** 
> - New here? Start with [HOOK-QUICK-REFERENCE.md](HOOK-QUICK-REFERENCE.md)
> - Want to test? See [HOOK-TESTING-PROMPTS.md](HOOK-TESTING-PROMPTS.md)
> - System setup? Check [.github/README.md](../.github/README.md)
> - Source of truth? Read [MidWay_project_outline.md](MidWay_project_outline.md), [session_handoff.md](session_handoff.md), [notes.md](notes.md)

## What This Hook Does

This hook **enforces your Phase 1 scope and validates all outputs** with data-driven checks. It operates in two stages:

1. **PreToolUse (Before):** Enforces Phase 1 scope
   - No Phase 2/3 work (checkpoint policy, graph rewriting, mu-TWO) starts until Phase 1 profiling is validated
   - Conflicts are caught early — if a command violates project docs, you get prompted to discuss instead of proceeding
   - Source of truth is enforced — project documentation (docs/MidWay_project_outline.md, docs/notes.md, docs/session_handoff.md) are the reference

2. **PostToolUse (After):** Validates all outputs with evidence
   - Python files must compile without syntax errors
   - JSON files must be valid and parseable
   - Shell scripts must have correct syntax
   - Benchmark runs must produce all expected plots and diagnostics JSON
   - Terminal commands must exit with code 0
   - **Result:** No silent failures; every output is verified

---

## How It Works

**Trigger:** Every time you ask me to modify code or run a terminal command, this hook runs first.

**Validation Logic:**

| Command Type | Phase 1 Approved | Blocked / Gated |
|---|---|---|
| **Code Modifications** | Edits to `benchmarks.py`, `graph_prof.py`, `graph_tracer.py` | Edits to `mu_two_policy.py`, anything with "checkpoint policy", "graph rewrite", "mu-TWO" |
| **Profiling Runs** | `python src/benchmarks.py` | Anything running Phase 2 policy or Phase 3 rewriting |
| **Diagnostic Output** | Plots, JSON diagnostics, classification reports | |
| **Git/Dev Commands** | `git`, `pip`, `python`, `conda`, `py_compile` | Commands that activate Phase 2/3 workflow |

**Outcomes:**

- ✅ **Allow** → Command runs immediately
- ⚠️ **Ask** → I prompt you: "This conflicts with Phase 1. Do you want to discuss why this is needed?"
- ❌ **Deny** → Command blocked; you must explicitly override

---

## Examples

### ✅ This Will Be Allowed

```
"Add a new diagnostic plot to show PARAM growth over time"
→ Allowed: Phase 1 visualization ✓
```

```
"Run python src/benchmarks.py ResNet152 4"
→ Allowed: Phase 1 profiling ✓
```

```
"Fix the classification of backward-pass temporaries in graph_prof.py"
→ Allowed: Phase 1 profiler improvement ✓
```

---

### ⚠️ This Will Trigger "Ask"

```
"Implement the mu-TWO checkpoint selection algorithm"
→ Ask: Phase 2 work blocked until Phase 1 validates OTHER memory ⚠️
```

```
"Rewrite the graph to insert recomputation subgraphs"
→ Ask: Phase 3 work not yet started ⚠️
```

```
"Modify mu_two_policy.py to add a new checkpoint heuristic"
→ Ask: Phase 2 file locked until Phase 1 decision ⚠️
```

```
"Add checkpoint validation to benchmarks.py"
→ Ask: Ambiguous—if it's about Phase 1 diagnostics, allowed; if Phase 2 policy, ask ⚠️
```

---

## Why This Matters (Harvard Perspective)

You're a professor working on a research-quality system. **Scope discipline is reputation protection:**

- **Prematurely activating Phase 2** → You end up debugging two phases at once, results are muddy
- **Conflicting with project docs** → Artifacts don't match scope, peer review suffers
- **Ad-hoc code changes** → Hard to trace when a decision was made or why
- **Clear scope → Clear evidence** → Solid foundation for the paper

---

## How to Override (If Needed)

If you intentionally need to work on Phase 2/3 before Phase 1 closes (edge case), tell me explicitly:

```
"I need to start Phase 2 checkpoint policy work now. This is intentional because [reason]."
```

I'll ask you to confirm, then proceed. But I'll also update the hook decision log so we can trace why the scope changed.

---

## What Gets Checked

**File modifications (`create_file`, `replace_string_in_file`):**
- Blocked files: `mu_two_policy.py`, `checkpoint.*`, `graph.*transform`
- User message keywords: "policy", "mu-TWO", "recompute", "rewrite", etc.
- Allowed files: Phase 1 profilers and tracer

**Terminal commands (`run_in_terminal`):**
- Blocked patterns: "policy", "checkpoint select", "mu-TWO", "graph rewrite"
- Allowed patterns: "benchmarks.py", "profile", "trace", git, pip, Python dev commands

**Ambiguous requests:**
- If unclear whether Phase 1 or Phase 2, you get asked for clarification before code runs

---

## Data-Driven Validation (PostToolUse)

After every command completes, the hook validates outputs with hard evidence:

| Output Type | Validation | Success Criteria |
|---|---|---|
| **Python files** | Syntax check via `py_compile` | File compiles without errors |
| **JSON files** | Parseable via `jq` | Valid JSON structure |
| **Shell scripts** | Syntax check via `bash -n` | No syntax errors |
| **Benchmark runs** | Artifact existence + validity | All plots exist + diagnostics JSON valid |
| **Terminal commands** | Exit code check | Exit code == 0 |

**Examples:**

```
You: "Run python src/benchmarks.py Bert 4"
→ Hook checks: plots/classification_diagnostics_Bert_bs4.json exists?
  plots/memory_growth_Bert_bs4.png exists?
  plots/memory_components_Bert_bs4.png exists?
  → If any missing or invalid JSON: BLOCK and report what failed
  → If all present and valid: CONTINUE with evidence snapshot
```

```
You: "Add a new diagnostic function to graph_prof.py"
→ Hook checks: graph_prof.py compiles?
  → If syntax error: BLOCK and show error line
  → If valid: CONTINUE with "Python file compiles successfully"
```

```
You: "Update the phase scope hook configuration"
→ Hook checks: .github/hooks/*.json is valid?
  → If malformed: BLOCK
  → If valid: CONTINUE
```

**Why This Matters:**

- **No silent failures** — If a benchmark crashes mid-way and creates incomplete files, you know immediately
- **Reproducibility** — Every artifact is verified before being used for the next step
- **Research integrity** — You can point to the exact moment each output was validated
- **Safety** — Catches syntax errors or malformed JSON before downstream tools consume them

---

## Configuration Files

- **Hook logic:** `.github/hooks/project-scope-validation.json`
- **PreToolUse validation:** `.github/scripts/validate-phase1-scope.sh`
- **PostToolUse validation:** `.github/scripts/validate-success-data.sh`
- **Source of truth:** `docs/MidWay_project_outline.md`, `docs/session_handoff.md`, `docs/notes.md`

---

## Next Steps

1. The hook is **active now** and validates every command I suggest
2. When Phase 1 closes (OTHER memory question answered), we update the hook to allow Phase 2
3. As you move to Phase 2, Phase 3 rules activate (similar pattern)

---

## Questions?

If a decision feels wrong or the hook is too strict/loose, let me know. I can refine the validation rules based on your feedback.

**Remember:** This is here to protect your work, not slow it down. Think of it as a research-quality checkpoint (pun intended).
