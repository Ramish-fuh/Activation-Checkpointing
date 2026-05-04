# Project Scope Validation Hook — Quick Reference

> **Navigation:**
> - Full design? See [HOOK-DESIGN.md](HOOK-DESIGN.md)
> - Test prompts? See [HOOK-TESTING-PROMPTS.md](HOOK-TESTING-PROMPTS.md)
> - System overview? See [.github/README.md](../.github/README.md)
> - Project scope? See [MidWay_project_outline.md](MidWay_project_outline.md)

## Two-Stage Validation

### Stage 1: PreToolUse (Before Command Runs)
Enforces Phase 1 scope against project documentation

### Stage 2: PostToolUse (After Command Completes)
Validates all outputs with hard evidence — no silent failures

---

## When You See This

### ✅ Approval Message
```
Allow: Code change approved. Aligns with Phase 1 scope.
Evidence: Python file compiles successfully.
```
→ Your request runs immediately. No delays.

---

### ✅ Success with Evidence
```
Continue: Benchmark completed. All plots and diagnostics JSON created and valid.
Evidence: 3 plots created, diagnostics JSON valid.
```
→ Outputs verified. Safe to use for next step.

---

### ⚠️ Conflict Detection
```
Ask: Detected Phase 2 work (checkpoint selection, graph rewriting). 
Current scope is Phase 1 profiling only. Please discuss.
```
→ I stop and ask: "Is this intentional? Why start Phase 2 now?"  
→ You can either:
   - **Clarify** why it's Phase 1 work → I proceed
   - **Discuss** why Phase 2 is needed → We update scope
   - **Cancel** the request → We don't write code

---

### ❌ Validation Failure
```
Block: File modification failed validation.
Reason: Python file has syntax errors at line 42.
```
→ Output rejected. Syntax error must be fixed before proceeding.

---

## What Gets Validated

| Operation | Before (PreToolUse) | After (PostToolUse) |
|-----------|---|---|
| Code modification | Phase 1 scope check | Syntax validation (Python/JSON/shell) |
| Terminal command | Scope + legitimacy check | Exit code + expected artifacts |
| Benchmark run | Scope check | All plots exist + JSON valid |
| File creation | Scope check | File exists + format validation |

---

## What Triggers Each Response

| You Say | Hook Response |
|---------|---------------|
| "Fix the peak memory calculation in graph_prof.py" | ✅ Allow + Python syntax check |
| "Add a new plot showing tensor-type growth" | ✅ Allow |
| "Run benchmarks on BERT" | ✅ Allow + artifact validation |
| "Implement the mu-TWO checkpoint policy" | ⚠️ Ask |
| "Rewrite the graph for activation checkpointing" | ⚠️ Ask |
| "Modify mu_two_policy.py to add heuristics" | ⚠️ Ask |
| "Optimize the profiler's liveness computation" | ✅ Allow + syntax check |

---

## Why Data-Driven Validation Matters

**Without it:**
- Benchmark runs silently fail → incomplete artifacts
- Python edits have hidden bugs → discovered much later
- JSON configs become malformed → hard to debug downstream
- Terminal commands fail → you don't know why

**With it:**
- ✅ Every output verified immediately
- ✅ Syntax errors caught before moving to next step
- ✅ Benchmarks guaranteed to produce complete artifacts
- ✅ Research-quality evidence trail

---

## Key Rules

1. **Phase 1 = Profiling + Diagnostics only**
   - Allowed: benchmarks.py, graph_prof.py, graph_tracer.py improvements
   - Blocked: mu_two_policy.py, checkpoint policy, graph rewriting

2. **Phase 2/3 work gets gated**
   - You can always override by saying "I want to start Phase 2 now because..."
   - But I'll ask you to confirm and document why

3. **All outputs are validated**
   - Python files must compile
   - JSON files must parse
   - Terminal commands must exit cleanly
   - Benchmark outputs must be complete

---

## When Phase 1 Closes

Once the `OTHER` memory question is answered and Phase 1 diagnostics validated:
- Update `.github/hooks/project-scope-validation.json`
- Activate Phase 2 rules (allow mu_two_policy.py edits)
- Block Phase 3 work until Phase 2 is complete

---

## Override Example

**If you need to bypass the hook:**

```
"I want to start Phase 2 checkpoint policy work now. 
Here's why: [reason]. Confirm before proceeding."
```

→ I'll ask for explicit confirmation, update the decision log, then proceed.

---

## Reference

- Full hook design: [`docs/HOOK-DESIGN.md`](docs/HOOK-DESIGN.md)
- Scope rules: [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- Validation script: [`.github/scripts/validate-phase1-scope.sh`](.github/scripts/validate-phase1-scope.sh)
- Hook config: [`.github/hooks/project-scope-validation.json`](.github/hooks/project-scope-validation.json)
