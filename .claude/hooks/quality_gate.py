#!/usr/bin/env python3
"""PostToolUse hook: ruff on edited libs/**.py, basedpyright on src only.

Scope rules (see spec 2026-07-17-agent-quality-rails-design.md):
  libs/*/src/**/*.py   -> ruff format+fix, C901 complexity, then basedpyright;
                          exit 2 on complexity or type errors
  libs/*/tests/**/*.py -> ruff format+fix and C901 complexity (exit 2 on it);
                          NEVER typechecked (TDD red phase writes imports for
                          code that does not exist yet)
  anything else        -> ignored (pyrightconfig.json has include: ["libs"])

Other ruff findings stay non-fatal: an unused import mid-edit is a normal
intermediate state. An over-complex function never is, so C901 blocks
(threshold: [tool.ruff.lint.mccabe] in the root pyproject.toml).

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
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_target(payload: object) -> tuple[Path, Path, Path] | None:
    """Return (repo root, absolute path, repo-relative path) for an edited libs/ .py file."""
    tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
    raw_path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if not isinstance(raw_path, str) or not raw_path or not project_dir:
        return None

    root = Path(project_dir)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    if path.suffix != ".py" or not path.exists():
        return None

    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None  # outside the repo

    # Require libs/<lib>/<src|tests>/...
    if len(rel.parts) < 3 or rel.parts[0] != "libs":
        return None
    return root, path, rel


def complexity_errors(root: Path, path: Path) -> list[str]:
    """Return C901 findings for the file (empty when clean or on tool failure)."""
    proc = run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--select",
            "C901",
            "--output-format",
            "concise",
            str(path),
        ],
        root,
    )
    if proc is None:
        return []
    return [f"  {line}" for line in proc.stdout.splitlines() if " C901 " in line]


def type_errors(root: Path, path: Path, rel: Path) -> list[str]:
    """Return basedpyright error lines for the file (empty when clean or unparseable)."""
    config = root / "pyrightconfig.json"
    if not config.exists():
        return []
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
        return []

    try:
        diagnostics = json.loads(proc.stdout)["generalDiagnostics"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []  # unparseable -> fail open

    return [
        format_diagnostic(rel, diag)
        for diag in diagnostics
        if isinstance(diag, dict) and diag.get("severity") == "error"
    ]


def format_diagnostic(rel: Path, diag: dict[str, object]) -> str:
    range_val = diag.get("range")
    start_val = range_val.get("start") if isinstance(range_val, dict) else None
    line = start_val.get("line", 0) if isinstance(start_val, dict) else 0
    rule = diag.get("rule", "")
    suffix = f" [{rule}]" if rule else ""
    return f"  {rel}:{line + 1} {diag.get('message', '')}{suffix}"


def main() -> int:
    try:
        target = resolve_target(json.load(sys.stdin))
        if target is None:
            return 0
        root, path, rel = target

        # Formatting and autofixable lint never block.
        run(["uv", "run", "ruff", "format", str(path)], root)
        run(["uv", "run", "ruff", "check", "--fix", str(path)], root)

        messages: list[str] = []
        complexity = complexity_errors(root, path)
        if complexity:
            messages += [f"ruff found {len(complexity)} over-complex function(s):"]
            messages += complexity
            messages += [
                (
                    "Split them into smaller functions (CI runs ruff with C901). "
                    "A `# noqa: C901` needs a written justification."
                )
            ]

        # tests: never typechecked. CI (make typecheck) DOES typecheck tests via
        # pyrightconfig include: ["libs"], so this gap is intentional and
        # one-directional: it just lets TDD red-phase tests through.
        types = type_errors(root, path, rel) if rel.parts[2] == "src" else []
        if types:
            messages += [f"basedpyright found {len(types)} error(s) in {rel}:"]
            messages += types
            messages += [
                "Fix these before continuing (CI runs basedpyright --level error)."
            ]

        if not messages:
            return 0
        print("\n".join(messages), file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 — fail open on any unexpected hook error (incl. bad JSON)
        return 0


if __name__ == "__main__":
    sys.exit(main())
