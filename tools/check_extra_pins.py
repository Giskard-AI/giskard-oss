"""Assert root pyproject lower bounds match workspace member versions.

Walks ``project.dependencies`` and ``project.optional-dependencies`` in the
repo-root ``pyproject.toml``. For every requirement that names a workspace
member, requires a ``>=`` lower bound equal to that member's ``version``.

Aggregator requirements that do not name a workspace member (for example
``giskard[full]``) are skipped.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
LIBS_DIR = REPO_ROOT / "libs"


def _member_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in sorted(LIBS_DIR.glob("*/pyproject.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        name = project.get("name")
        version = project.get("version")
        if not name or not version:
            raise SystemExit(
                f"missing project.name/version in {path.relative_to(REPO_ROOT)}"
            )
        versions[name] = version
    if not versions:
        raise SystemExit(f"no workspace members found under {LIBS_DIR}")
    return versions


def _lower_bound(specifiers: SpecifierSet) -> str | None:
    lowers = [spec.version for spec in specifiers if spec.operator == ">="]
    if len(lowers) != 1:
        return None
    return lowers[0]


def _iter_root_requirements(data: dict) -> list[tuple[str, str]]:
    """Return (location, requirement_string) pairs from root project deps."""
    project = data.get("project") or {}
    entries: list[tuple[str, str]] = []
    for req in project.get("dependencies") or []:
        entries.append(("project.dependencies", req))
    optional = project.get("optional-dependencies") or {}
    for extra, reqs in optional.items():
        for req in reqs or []:
            entries.append((f"project.optional-dependencies.{extra}", req))
    return entries


def main() -> int:
    data = tomllib.loads(ROOT_PYPROJECT.read_text(encoding="utf-8"))
    members = _member_versions()
    mismatches: list[str] = []

    for location, req_str in _iter_root_requirements(data):
        try:
            req = Requirement(req_str)
        except InvalidRequirement as exc:
            mismatches.append(f"{location}: invalid requirement {req_str!r}: {exc}")
            continue

        expected = members.get(req.name)
        if expected is None:
            continue

        found = _lower_bound(req.specifier)
        if found is None:
            mismatches.append(
                f"{location}: {req_str!r} names workspace member {req.name!r} "
                f"but has no single '>=' lower bound (expected >={expected})"
            )
            continue
        if found != expected:
            mismatches.append(
                f"{location}: {req.name} lower bound is {found!r}, "
                f"expected {expected!r} (from libs member version)"
            )

    if mismatches:
        print(
            "Root extra/dependency pins drift from workspace member versions:",
            file=sys.stderr,
        )
        for line in mismatches:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"OK: {len(members)} workspace members; root lower bounds match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
