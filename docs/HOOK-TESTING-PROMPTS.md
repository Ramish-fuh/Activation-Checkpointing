# Hook Testing Prompts

> **Navigation:**
> - Quick reference? See [HOOK-QUICK-REFERENCE.md](HOOK-QUICK-REFERENCE.md)
> - Full design? See [HOOK-DESIGN.md](HOOK-DESIGN.md)
> - System overview? See [.github/README.md](../.github/README.md)

Use these prompts to test the scope validation and data-driven validation hooks.

---

## Test 1: Phase 1 Allowed ✅
**Expected:** PreToolUse allows, PostToolUse validates artifact

```
Run this command: python src/benchmarks.py Bert 4

Then ask me to:
1. Inspect the plots/classification_diagnostics_Bert_bs4.json file
2. Check if the accounting invariants passed
```

**What you'll see:**
- ✅ PreToolUse: Allow (Phase 1 profiling approved)
- ✅ PostToolUse: All 3 plots exist + JSON is valid

---

## Test 2: Phase 2 Gated ⚠️
**Expected:** PreToolUse asks for discussion, no code written

```
"Implement the mu-TWO checkpoint selection algorithm based on the profiler output."
```

**What you'll see:**
- ⚠️ PreToolUse: Ask (Phase 2 work detected)
- I'll stop and ask: "This violates Phase 1 scope. Is this intentional?"
- No code modifications happen until you clarify

---

## Test 3: Data Validation (Python Syntax) ✅
**Expected:** PreToolUse allows, PostToolUse validates syntax

```
"Add a new debug function to graph_prof.py that prints tensor counts."
```

**What you'll see:**
- ✅ PreToolUse: Allow (Phase 1 profiler improvement)
- ✅ PostToolUse: Python file compiles successfully

---

## Test 4: Data Validation (JSON) ✅
**Expected:** PreToolUse allows, PostToolUse validates JSON structure

```
"Update the hook configuration to add a timeout of 15 seconds for PostToolUse."
```

**What you'll see:**
- ✅ PreToolUse: Allow (configuration file)
- ✅ PostToolUse: JSON is valid

---

## Test 5: Ambiguous Request ⚠️
**Expected:** PreToolUse asks for clarification

```
"Improve the benchmark harness to better detect when checkpointing is active."
```

**What you'll see:**
- ⚠️ PreToolUse: Ask (could be Phase 1 profiling or Phase 2 checkpoint logic)
- I'll ask: "Is this for Phase 1 profiling accuracy, or Phase 2 checkpoint selection?"
- Once you clarify, I'll proceed

---

## Test 6: File Doesn't Exist ❌
**Expected:** PostToolUse blocks if artifact missing

```
"Create a new file called docs/test_hook.md with the content: 'Testing the hook system.'"
```

**Then ask:**
```
"Now delete that file and run a benchmark again to show PostToolUse catches missing artifacts."
```

**What you'll see:**
- ✅ PreToolUse: Allow (Phase 1 diagnostic file)
- ❌ PostToolUse: Block if key plots are missing

---

## Test 7: Phase 2 File Locked ⚠️
**Expected:** PreToolUse asks before allowing mu_two_policy.py edits

```
"Fix the checkpoint selection logic in src/mu_two_policy.py to handle edge cases."
```

**What you'll see:**
- ⚠️ PreToolUse: Ask (mu_two_policy.py is Phase 2, locked during Phase 1)
- I'll stop: "Phase 2 file locked until Phase 1 validation completes. Discuss if needed."

---

## Recommended Test Sequence

1. **Run Test 1** first (happy path — see everything working)
2. **Run Test 2** (confirm Phase 2 is blocked)
3. **Run Test 3** (see syntax validation working)
4. **Run Test 4** (see JSON validation working)
5. **Run Test 5** (see ambiguity handling)

This gives you full coverage of both PreToolUse and PostToolUse stages.

---

## How to Run a Test

Simply copy-paste one of the test prompts above directly into the chat. For example:

```
Run this command: python src/benchmarks.py Bert 4
```

Then watch for the hook responses:
- **PreToolUse output:** Appears before execution
- **Command output:** Appears during execution
- **PostToolUse output:** Appears after execution with validation results

---

## Understanding the Results

| Result | Means | Next Step |
|---|---|---|
| ✅ Allow → Continue | Scope OK, validation passed | Safe to proceed |
| ⚠️ Ask → Discussion | Potential scope conflict | Clarify intent before code runs |
| ❌ Block → Failed | Validation failed (syntax, artifact missing) | Fix issue before retry |

---

## Example: Full Test Walkthrough

**You send:**
```
Run this command: python src/benchmarks.py Bert 4
```

**I respond:**

```
PreToolUse: ✅ Allow
Reason: Benchmark profiling is Phase 1 approved.

[Running benchmark...]

PostToolUse: ✅ Continue
Evidence: Benchmark completed. All plots created and diagnostics JSON is valid.
Artifacts verified:
- plots/classification_diagnostics_Bert_bs4.json ✓
- plots/memory_growth_Bert_bs4.png ✓
- plots/memory_components_Bert_bs4.png ✓
```

---

## Need Help Interpreting Results?

If you get an unexpected response:
1. Check the hook reason message
2. Refer to [docs/HOOK-DESIGN.md](docs/HOOK-DESIGN.md) for detailed validation logic
3. Check [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) for examples
4. Ask me to explain the decision
