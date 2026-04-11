#!/usr/bin/env python3
import json
import pathlib
import sys

WORKSPACE_ROOT = pathlib.Path("/Users/shuramish/Documents/Harvard/Big Data systems /project")
POLICY_MESSAGE = (
    "Keep the implementation paper-faithful to μ-TWO. Do not deviate from the "
    "paper or the assignment requirements. Treat the paper and project docs as "
    "the source of truth for all future requests."
)


def _collect_log_files() -> list[pathlib.Path]:
    logs = []
    for pattern in ("Model: *.txt", "*Resnet*.txt", "*Bert*.txt"):
        for path in sorted(WORKSPACE_ROOT.glob(pattern)):
            if path not in logs:
                logs.append(path)
    return logs


def _collect_original_files() -> list[pathlib.Path]:
    """Collect original source files for comparison/verification."""
    original_dir = WORKSPACE_ROOT / "original"
    if not original_dir.exists():
        return []
    return sorted(original_dir.glob("*.py"))


def _summarize_log(path: pathlib.Path) -> str:
    try:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
    except Exception as exc:  # pragma: no cover - hook should fail soft
        return f"Read file: {path} (error reading: {exc})"
    return f"Read file: {path} ({line_count} lines)"


def _summarize_original_file(path: pathlib.Path) -> str:
    try:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
    except Exception as exc:  # pragma: no cover - hook should fail soft
        return f"{path.name} (error reading: {exc})"
    return f"{path.name} ({line_count} lines)"


def main() -> None:
    if not sys.stdin.isatty():
        _ = sys.stdin.read()
    logs = _collect_log_files()
    originals = _collect_original_files()
    
    log_summary = "\n".join(f"- {_summarize_log(path)}" for path in logs)
    original_summary = "\n".join(f"- {_summarize_original_file(path)}" for path in originals)

    message = POLICY_MESSAGE
    if log_summary or original_summary:
        parts = [
            "Read file before any code-related tool use. "
            "The hook has reviewed the available context:\n"
        ]
        if log_summary:
            parts.append("Benchmark logs (authoritative debugging context):")
            parts.append(log_summary)
        if original_summary:
            parts.append(
                "\nOriginal code (for verification/comparison - was it changed?):"
            )
            parts.append(f"See /original directory with files:\n{original_summary}")
        parts.append(f"\n{POLICY_MESSAGE}")
        message = "\n".join(parts)
    else:
        message = (
            "Read file before any code-related tool use. No benchmark logs or "
            "original code found for context.\n\n"
            f"{POLICY_MESSAGE}"
        )

    marker = WORKSPACE_ROOT / ".github" / "hooks" / ".policy_hook_ran"
    marker.write_text("reviewed\n", encoding="utf-8")

    print(json.dumps({"continue": True, "systemMessage": message}))


if __name__ == "__main__":
    main()
