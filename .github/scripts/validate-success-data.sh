#!/bin/bash
# Data-driven validation: verify command outputs exist and are valid
#
# Source of Truth:
#   - Expected outputs: src/benchmarks.py (graph_transformation function)
#   - Classification: src/graph_prof.py (tensor role definitions)
#   - Design docs: docs/HOOK-DESIGN.md
#
# Input: JSON on stdin with tool result and context
# Output: JSON with decision (continue|block) based on artifact validation

set -e

INPUT=$(cat)

# Extract context
TOOL=$(echo "$INPUT" | jq -r '.toolName // "unknown"')
EXIT_CODE=$(echo "$INPUT" | jq -r '.exitCode // 1')
STDOUT=$(echo "$INPUT" | jq -r '.stdout // ""')
STDERR=$(echo "$INPUT" | jq -r '.stderr // ""')
FILE_PATH=$(echo "$INPUT" | jq -r '.filePath // ""')
CMD=$(echo "$INPUT" | jq -r '.command // ""')

# === VALIDATION RULES ===

case "$TOOL" in
  "create_file" | "replace_string_in_file")
    # File modification: verify file exists and is syntactically valid
    
    if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
      echo "{
        \"decision\": \"block\",
        \"reason\": \"File not found after modification: $FILE_PATH\"
      }"
      exit 0
    fi
    
    # Validate Python files syntax
    if [[ "$FILE_PATH" == *.py ]]; then
      if python3 -m py_compile "$FILE_PATH" 2>/dev/null; then
        echo "{
          \"decision\": \"continue\",
          \"evidence\": \"Python file compiles successfully\",
          \"file\": \"$FILE_PATH\"
        }"
      else
        echo "{
          \"decision\": \"block\",
          \"reason\": \"Python file has syntax errors\",
          \"file\": \"$FILE_PATH\"
        }"
      fi
      exit 0
    fi
    
    # Validate JSON files
    if [[ "$FILE_PATH" == *.json ]]; then
      if jq empty "$FILE_PATH" 2>/dev/null; then
        echo "{
          \"decision\": \"continue\",
          \"evidence\": \"JSON is valid\",
          \"file\": \"$FILE_PATH\"
        }"
      else
        echo "{
          \"decision\": \"block\",
          \"reason\": \"JSON is malformed\",
          \"file\": \"$FILE_PATH\"
        }"
      fi
      exit 0
    fi
    
    # Validate shell scripts
    if [[ "$FILE_PATH" == *.sh ]]; then
      if bash -n "$FILE_PATH" 2>/dev/null; then
        echo "{
          \"decision\": \"continue\",
          \"evidence\": \"Shell script syntax is valid\",
          \"file\": \"$FILE_PATH\"
        }"
      else
        echo "{
          \"decision\": \"block\",
          \"reason\": \"Shell script has syntax errors\",
          \"file\": \"$FILE_PATH\"
        }"
      fi
      exit 0
    fi
    
    # Markdown/YAML files: just verify they exist
    if [[ "$FILE_PATH" == *.md ]] || [[ "$FILE_PATH" == *.yaml ]] || [[ "$FILE_PATH" == *.yml ]]; then
      echo "{
        \"decision\": \"continue\",
        \"evidence\": \"File created/modified\",
        \"file\": \"$FILE_PATH\"
      }"
      exit 0
    fi
    
    # Default: allow if file exists
    echo "{
      \"decision\": \"continue\",
      \"evidence\": \"File exists\",
      \"file\": \"$FILE_PATH\"
    }"
    exit 0
    ;;
    
  "run_in_terminal")
    # Terminal command: check for expected outputs
    
    if [ "$EXIT_CODE" != "0" ]; then
      echo "{
        \"decision\": \"block\",
        \"reason\": \"Command exited with code $EXIT_CODE\",
        \"stderr\": \"$STDERR\"
      }"
      exit 0
    fi
    
    # Benchmark runs: verify plots and JSON were created
    if echo "$CMD" | grep -i "benchmarks.py" > /dev/null; then
      # Extract model and batch size from command
      MODEL=$(echo "$CMD" | awk '{print $NF}' | head -1)
      BS=$(echo "$CMD" | awk '{print $NF}' | tail -1)
      
      EXPECTED_FILES=(
        "plots/classification_diagnostics_${MODEL}_bs${BS}.json"
        "plots/memory_growth_${MODEL}_bs${BS}.png"
        "plots/memory_components_${MODEL}_bs${BS}.png"
      )
      
      MISSING=()
      for f in "${EXPECTED_FILES[@]}"; do
        if [ ! -f "$f" ]; then
          MISSING+=("$f")
        fi
      done
      
      if [ ${#MISSING[@]} -gt 0 ]; then
        echo "{
          \"decision\": \"block\",
          \"reason\": \"Benchmark completed but expected artifacts missing\",
          \"missing\": $(echo "${MISSING[@]}" | jq -R 'split(" ")')
        }"
        exit 0
      fi
      
      # Verify JSON is valid
      if jq empty "plots/classification_diagnostics_${MODEL}_bs${BS}.json" 2>/dev/null; then
        echo "{
          \"decision\": \"continue\",
          \"evidence\": \"Benchmark completed. All plots and diagnostics JSON created and valid.\",
          \"artifacts\": {
            \"plots\": 3,
            \"diagnostics\": \"valid JSON\"
          }
        }"
      else
        echo "{
          \"decision\": \"block\",
          \"reason\": \"Diagnostics JSON is malformed\",
          \"file\": \"plots/classification_diagnostics_${MODEL}_bs${BS}.json\"
        }"
      fi
      exit 0
    fi
    
    # Git/Python/package commands: check exit code
    if echo "$CMD" | grep -iE "git|pip|conda|python|py_compile" > /dev/null; then
      echo "{
        \"decision\": \"continue\",
        \"evidence\": \"Command executed successfully (exit code 0)\"
      }"
      exit 0
    fi
    
    # Generic success check
    echo "{
      \"decision\": \"continue\",
      \"evidence\": \"Command completed successfully\"
    }"
    exit 0
    ;;
    
  *)
    # Unknown tool: pass through if successful
    if [ "$EXIT_CODE" = "0" ]; then
      echo "{
        \"decision\": \"continue\",
        \"evidence\": \"Operation completed successfully\"
      }"
    else
      echo "{
        \"decision\": \"block\",
        \"reason\": \"Operation failed with exit code $EXIT_CODE\"
      }"
    fi
    exit 0
    ;;
esac
