"""Validate inline message templates from the command line.

Reads a JSON array of template strings on stdin and writes a JSON array of
results to stdout, one item per template, in order:
``{"ok": true}`` or ``{"ok": false, "error": "<message>", "line": <int|null>}``.

Exit code 0 when validation ran (whatever the results), 2 when stdin is not a
JSON array of strings.
"""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from giskard.agents.templates import TemplateValidationError, validate_template


def _package_version() -> str:
    try:
        return version("giskard-agents")
    except PackageNotFoundError:
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m giskard.agents.templates.validate",
        description="Validate templates (JSON array of strings on stdin).",
    )
    parser.add_argument("--variables", nargs="+", required=True, metavar="NAME")
    parser.add_argument("--required", nargs="*", default=[], metavar="NAME")
    parser.add_argument(
        "--version", action="version", version=f"giskard-agents {_package_version()}"
    )
    return parser


def _check(source: str, variables: list[str], required: list[str]) -> dict[str, Any]:
    try:
        validate_template(source, variables=variables, required=required)
    except TemplateValidationError as exc:
        return {"ok": False, "error": exc.message, "line": exc.lineno}
    return {"ok": True}


def main() -> int:
    args = _parser().parse_args()
    try:
        templates = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(f"stdin is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(templates, list) or not all(
        isinstance(item, str) for item in templates
    ):
        print("stdin must be a JSON array of strings", file=sys.stderr)
        return 2
    results = [_check(t, args.variables, args.required) for t in templates]
    json.dump(results, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
