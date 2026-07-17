#!/usr/bin/env python3
"""PreToolUse hook: require `uv run` for python/pytest invocations.

Deliberately narrow: only anchored, bare invocations are blocked. Compound
forms (`cd x && pytest`, `PYTHONPATH=. python f.py`) are NOT caught -- a
cleverer regex would false-block legitimate commands like `echo "run pytest"`.
A guard that is occasionally silent beats one that blocks valid work.
"""

import json
import re
import sys

# Anchored at string start, optional leading whitespace only.
BARE = re.compile(r"^\s*(python3?|pytest)\b")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # fail open on unparseable input

    command = payload.get("tool_input", {}).get("command", "")
    if not isinstance(command, str):
        return 0

    match = BARE.match(command)
    if not match:
        return 0

    tool = match.group(1)
    print(
        f"Blocked: bare `{tool}` fails with ModuleNotFoundError in this repo.\n"
        f"Use `uv run {command.strip()}` instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
