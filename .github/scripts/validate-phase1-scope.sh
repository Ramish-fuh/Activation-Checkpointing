#!/bin/bash
# Validate that commands/code changes align with CS265 Phase 1 scope
# 
# Source of Truth:
#   - Phase scope: docs/MidWay_project_outline.md
#   - Current status: docs/session_handoff.md
#   - Design docs: docs/HOOK-DESIGN.md
#
# Input: JSON on stdin with tool info and user request context
# Output: JSON with permissionDecision (allow|ask|deny) or block

set -e

# Read input JSON from stdin
INPUT=$(cat)

# Extract key fields
TOOL=$(echo "$INPUT" | jq -r '.toolName // "unknown"')
USER_MESSAGE=$(echo "$INPUT" | jq -r '.userMessage // ""' | tr '[:upper:]' '[:lower:]')
FILE_PATH=$(echo "$INPUT" | jq -r '.filePath // ""')

# === PHASE 1 SCOPE DEFINITION ===
# ALLOWED: profiling, tracing, diagnostics, plotting, JSON output, tensor classification
# BLOCKED: checkpoint policy selection, graph rewriting, mu-TWO algorithm, FX graph transformation
# GATED: modifications to Phase 1 architecture (requires discussion)

# Phase 2/3 keywords that indicate out-of-scope work
PHASE2_KEYWORDS="checkpoint.*policy|mu.*two|checkpoint.*selection|retain.*activation|recompute|graph.*rewrite|policy.*config"
PHASE3_KEYWORDS="apply.*checkpoint|graph.*transform|insert.*recomputation|rewrite.*backward"

# Phase 1 approved keywords
PHASE1_KEYWORDS="profile|trace|diagnostic|plot|classify|tensor|liveness|memory.*break|peak.*memory|visualization|benchmark|OTHER"

# Blocked file modifications
BLOCKED_FILES="mu_two_policy|checkpoint.*plan|graph.*transform"

# === TOOL-SPECIFIC RULES ===

case "$TOOL" in
  "create_file" | "replace_string_in_file")
    # Code modifications: cross-check file and intent
    
    # Check if trying to modify Phase 2/3 files
    if echo "$FILE_PATH" | grep -iE "$BLOCKED_FILES" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"ask\",
          \"permissionDecisionReason\": \"Attempting to modify Phase 2 checkpoint policy file (mu_two_policy.py). Phase 1 scope is profiling and diagnostics only. Please confirm this is needed before Phase 1 validation is complete.\"
        }
      }"
      exit 0
    fi
    
    # Check user message for Phase 2/3 intent
    if echo "$USER_MESSAGE" | grep -iE "$PHASE2_KEYWORDS|$PHASE3_KEYWORDS" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"ask\",
          \"permissionDecisionReason\": \"Detected Phase 2/3 work (checkpoint selection, graph rewriting, or mu-TWO policy). Current scope is Phase 1 profiling only. The OTHER memory classification must be validated first. Please discuss the conflict.\"
        }
      }"
      exit 0
    fi
    
    # Phase 1 files: allow
    if echo "$FILE_PATH" | grep -iE "benchmarks|graph_prof|graph_tracer|activation_checkpoint" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"allow\"
        }
      }"
      exit 0
    fi
    
    # Diagnostic/plotting modifications: allow
    if echo "$USER_MESSAGE" | grep -iE "$PHASE1_KEYWORDS" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"allow\"
        }
      }"
      exit 0
    fi
    
    # Ambiguous: ask for clarification
    echo "{
      \"hookSpecificOutput\": {
        \"hookEventName\": \"PreToolUse\",
        \"permissionDecision\": \"ask\",
        \"permissionDecisionReason\": \"Unclear if this aligns with Phase 1 scope (profiling + diagnostics). Please clarify: is this for improving the profiler, visualization, or tensor classification?\"
      }
    }"
    exit 0
    ;;
    
  "run_in_terminal")
    # Terminal commands: check for destructive or Phase 2 operations
    CMD=$(echo "$INPUT" | jq -r '.command // ""' | tr '[:upper:]' '[:lower:]')
    
    # Allow benchmark runs and profiling
    if echo "$CMD" | grep -iE "benchmarks.py|profile|trace" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"allow\"
        }
      }"
      exit 0
    fi
    
    # Block policy or rewrite commands
    if echo "$CMD" | grep -iE "policy|checkpoint.*select|mu.*two|graph.*rewrite" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"ask\",
          \"permissionDecisionReason\": \"Detected Phase 2/3 command (policy selection or graph rewriting). Phase 1 scope is diagnostics only. Verify this is needed.\"
        }
      }"
      exit 0
    fi
    
    # Allow git, pip, and dev commands
    if echo "$CMD" | grep -iE "git|pip|python|py_compile|conda" > /dev/null 2>&1; then
      echo "{
        \"hookSpecificOutput\": {
          \"hookEventName\": \"PreToolUse\",
          \"permissionDecision\": \"allow\"
        }
      }"
      exit 0
    fi
    ;;
    
  *)
    # Other tools: allow by default
    echo "{
      \"hookSpecificOutput\": {
        \"hookEventName\": \"PreToolUse\",
        \"permissionDecision\": \"allow\"
      }
    }"
    exit 0
    ;;
esac

# Fallback: allow unknown
echo "{
  \"hookSpecificOutput\": {
    \"hookEventName\": \"PreToolUse\",
    \"permissionDecision\": \"allow\"
  }
}"
