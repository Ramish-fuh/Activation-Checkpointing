# CS265 Scope Validation & Data-Driven Hook System

**Purpose:** Enforce Phase 1 scope and validate all outputs for research-grade quality. No setup needed—just commit and use.

---

## 📋 Complete Documentation Index

### User-Facing Guides
| Document | Purpose | Read This If... |
|---|---|---|
| [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) | 2-minute reference | You want to understand what the hook does |
| [docs/HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md) | Copy-paste test prompts | You want to test the hook system |
| [docs/HOOK-DESIGN.md](docs/HOOK-DESIGN.md) | Deep design reference | You want full technical details |

### Project Scope Documents (Source of Truth)
| Document | Defines |
|---|---|
| [docs/MidWay_project_outline.md](docs/MidWay_project_outline.md) | Phase 1/2/3 scope, problems, technical approach |
| [docs/session_handoff.md](docs/session_handoff.md) | Current phase status, handoff context, next steps |
| [docs/notes.md](docs/notes.md) | Running context, model specs, project overview |

### Hook Implementation Files
| File | Role |
|---|---|
| [.github/hooks/project-scope-validation.json](.github/hooks/project-scope-validation.json) | Hook configuration (PreToolUse + PostToolUse) |
| [.github/scripts/validate-phase1-scope.sh](.github/scripts/validate-phase1-scope.sh) | PreToolUse: Scope enforcement logic |
| [.github/scripts/validate-success-data.sh](.github/scripts/validate-success-data.sh) | PostToolUse: Output artifact validation |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Agent-level instructions (auto-loaded by VS Code) |

---

## 🚀 Quick Start (New System)

1. **Copy these files to your new system:**
   ```
   .github/hooks/
   .github/scripts/
   .github/copilot-instructions.md
   docs/HOOK-*.md
   ```

2. **No installation needed** — hooks auto-load in VS Code

3. **Test immediately:**
   - Open [docs/HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md)
   - Copy one test prompt into the chat
   - Observe PreToolUse + PostToolUse validation

4. **Reference as you work:**
   - Quick lookup: [docs/HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md)
   - Full details: [docs/HOOK-DESIGN.md](docs/HOOK-DESIGN.md)

---

## 📊 How It Works (30-Second Overview)

### PreToolUse (Before Code Runs)
Validates against project scope:
```
You: "Implement checkpoint policy"
Hook: ⚠️ Ask - Phase 2 work detected. Phase 1 only. Discuss?
Result: No code written until you clarify
```

### PostToolUse (After Code Runs)
Validates all outputs with hard evidence:
```
You: "Run benchmarks"
Hook: ✅ Continue - All 3 plots created, JSON valid
Result: Safe to use artifacts for next step
```

---

## 🔗 Cross-References

**All validation decisions reference:**
- Phase scope: `docs/MidWay_project_outline.md` (Problems 1-5, Phase definitions)
- Current status: `docs/session_handoff.md` (Current Decision section)
- Running context: `docs/notes.md` (Model specs, phase breakdown)

**All artifact validation checks:**
- Expected outputs: `src/benchmarks.py` (graph_transformation function)
- Classification logic: `src/graph_prof.py` (PARAM/GRAD/ACT/OPT/OTHER definitions)
- Tracing approach: `src/graph_tracer.py` (SEPFunction markers)

---

## ✅ What Gets Validated

### PreToolUse (Scope Check)
- ✅ Phase 1 work (profiling, plotting, classification)
- ⚠️ Phase 2 work (checkpoint policy, selection)
- ⚠️ Phase 3 work (graph rewriting)
- ❌ Modifications to `mu_two_policy.py` during Phase 1

### PostToolUse (Data Check)
- ✅ Python files compile (`py_compile`)
- ✅ JSON files parse (`jq empty`)
- ✅ Shell scripts validate (`bash -n`)
- ✅ Benchmark artifacts exist (3 plots + JSON)
- ✅ Terminal commands exit cleanly (code 0)

---

## 📁 File Locations Reference

```
.github/
├── copilot-instructions.md          ← Agent instructions (auto-loaded)
├── hooks/
│   └── project-scope-validation.json ← Hook config (PreToolUse + PostToolUse)
└── scripts/
    ├── validate-phase1-scope.sh      ← PreToolUse enforcement
    └── validate-success-data.sh      ← PostToolUse validation

docs/
├── HOOK-QUICK-REFERENCE.md          ← Start here (2 min read)
├── HOOK-TESTING-PROMPTS.md          ← Copy-paste test prompts
├── HOOK-DESIGN.md                   ← Full technical reference
├── MidWay_project_outline.md        ← Source of truth: Phase scope
├── session_handoff.md               ← Source of truth: Current status
└── notes.md                         ← Source of truth: Running context
```

---

## 🎯 For New Systems: Minimal Overhead

The hook system is **self-contained**:
- ✅ No package installations
- ✅ No environment setup
- ✅ No configuration files to edit
- ✅ No system dependencies beyond `bash`, `jq`, `python3`
- ✅ All scope rules embedded in scripts
- ✅ All references point to committed docs

**Ready to commit and push.** It will work on any machine with the repo cloned.

---

## 📞 Need Help?

| Question | See |
|---|---|
| What does the hook do? | [HOOK-QUICK-REFERENCE.md](docs/HOOK-QUICK-REFERENCE.md) |
| How do I test it? | [HOOK-TESTING-PROMPTS.md](docs/HOOK-TESTING-PROMPTS.md) |
| Full technical details? | [HOOK-DESIGN.md](docs/HOOK-DESIGN.md) |
| What's Phase 1 scope? | [MidWay_project_outline.md](docs/MidWay_project_outline.md) |
| What's the current status? | [session_handoff.md](docs/session_handoff.md) |

---

**Last Updated:** May 3, 2026  
**Status:** ✅ Ready for production use  
**Commit:** All docs and scripts included
