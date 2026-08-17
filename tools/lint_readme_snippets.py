"""Fail when README fences use positional CheckResult.success/failure strings."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_FENCE_RE = re.compile(
    r"^```[ \t]*(?:python|py)[ \t]*\n(.*?)^```[ \t]*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_FORBIDDEN_CALLS = frozenset({"CheckResult.success", "CheckResult.failure"})


def _check_source(source: str, location: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        errors.append(f"{location}: invalid Python fence: {exc.msg}")
        return errors

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "CheckResult":
            continue
        if func.attr not in ("success", "failure"):
            continue
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.JoinedStr) or (
                isinstance(first, ast.Constant) and isinstance(first.value, str)
            ):
                errors.append(
                    f"{location}:{node.lineno}: positional CheckResult.{func.attr}(str) "
                    "is forbidden; use message="
                )
    return errors


def lint_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for match in _FENCE_RE.finditer(text):
        source = match.group(1)
        errors.extend(_check_source(source, str(path)))
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "README.md",
        *sorted((root / "libs").glob("*/README.md")),
    ]
    errors: list[str] = []
    for path in targets:
        if path.is_file():
            errors.extend(lint_markdown(path))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
