#!/usr/bin/env python3
"""PostToolUse hook: ruff on edited libs/**.py, basedpyright on src only.

Scope rules (see spec 2026-07-17-agent-quality-rails-design.md):
  libs/*/src/**/*.py   -> ruff format+fix, then basedpyright; exit 2 on errors
  libs/*/tests/**/*.py -> ruff only; NEVER typechecked (TDD red phase writes
                          imports for code that does not exist yet)
  anything else        -> ignored (pyrightconfig.json has include: ["libs"])

basedpyright finds pyrightconfig.json by walking up from CWD, not from the
target file, so --project is passed explicitly. Without it, a clean source
file reports bogus reportMissingImports and the hook would block every edit.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

TIMEOUT = 60


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run a command, returning None on any environmental failure (fail open)."""
    try:
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    try:
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0

        raw_path = tool_input.get("file_path")
        if not isinstance(raw_path, str):
            return 0

        project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
        if not raw_path or not project_dir:
            return 0

        root = Path(project_dir)
        config = root / "pyrightconfig.json"
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path

        if path.suffix != ".py" or not path.exists():
            return 0

        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            return 0  # outside the repo

        parts = rel.parts
        # Require libs/<lib>/<src|tests>/...
        if len(parts) < 3 or parts[0] != "libs":
            return 0

        # ruff runs on both src and tests; failures here are non-fatal.
        run(["uv", "run", "ruff", "format", str(path)], root)
        run(["uv", "run", "ruff", "check", "--fix", str(path)], root)

        if parts[2] != "src":
            # tests: ruff only, never typechecked. CI (make typecheck) DOES
            # typecheck tests via pyrightconfig include: ["libs"], so this
            # gap is intentional and one-directional: the hook never blocks
            # anything CI would pass, it just lets TDD red-phase tests through.
            return 0
        if not config.exists():
            return 0

        proc = run(
            [
                "uv",
                "run",
                "basedpyright",
                "--level",
                "error",
                "--project",
                str(config),
                "--outputjson",
                str(path),
            ],
            root,
        )
        if proc is None:
            return 0

        try:
            report = json.loads(proc.stdout)
            error_count = report["summary"]["errorCount"]
            diagnostics = report["generalDiagnostics"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0  # unparseable -> fail open

        if error_count == 0:
            return 0

        lines = [f"basedpyright found {error_count} error(s) in {rel}:"]
        for diag in diagnostics:
            if not isinstance(diag, dict) or diag.get("severity") != "error":
                continue
            range_val = diag.get("range") or {}
            start_val = range_val.get("start") or {}
            line_no = start_val.get("line", 0) + 1
            rule = diag.get("rule", "")
            suffix = f" [{rule}]" if rule else ""
            lines.append(f"  {rel}:{line_no} {diag.get('message', '')}{suffix}")
        lines.append(
            "Fix these before continuing (CI runs basedpyright --level error)."
        )
        print("\n".join(lines), file=sys.stderr)
        return 2
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
