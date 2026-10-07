"""Assert workspace-member pins match expected versions.

Default (no args): walks ``project.dependencies`` and
``project.optional-dependencies`` in the repo-root ``pyproject.toml``. For
every requirement that names a workspace member, requires a ``>=`` lower
bound equal to that member's ``version``.

With ``PACKAGE VERSION``: walks the same fields in the root file and every
``libs/*/pyproject.toml``. Every versioned PEP 508 requirement naming
``PACKAGE`` must have a single ``>=`` bound equal to ``VERSION``. Unversioned
extra aggregators (for example ``giskard-checks[all]``) are skipped. Exit 0
when no versioned requirement names ``PACKAGE`` (the ``giskard`` metapackage).

Aggregator requirements that do not name a workspace member (for example
``giskard[full]``) are skipped in the default mode.
"""

import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
LIBS_DIR = REPO_ROOT / "libs"

_DISALLOWED_OPS = frozenset({"==", "~=", ">", "==="})


def _canonical_name(name: str) -> str:
    return str(canonicalize_name(name))


def _member_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in sorted(LIBS_DIR.glob("*/pyproject.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        name = project.get("name")
        version = project.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise SystemExit(
                f"missing project.name/version in {path.relative_to(REPO_ROOT)}"
            )
        versions[_canonical_name(name)] = version
    if not versions:
        raise SystemExit(f"no workspace members found under {LIBS_DIR}")
    return versions


def _lower_bound(specifiers: SpecifierSet) -> str | None:
    lowers = [spec.version for spec in specifiers if spec.operator == ">="]
    if len(lowers) != 1:
        return None
    return lowers[0]


def _package_pin_reason(specifiers: SpecifierSet, expected: str) -> str | None:
    """Return a mismatch description, or None when the pin is valid."""
    ops = [spec.operator for spec in specifiers]
    if any(op in _DISALLOWED_OPS for op in ops):
        return (
            f"{str(specifiers)!r} uses ==/~=/> instead of a single "
            f"'>=' lower bound (expected >={expected})"
        )
    found = _lower_bound(specifiers)
    if found is None:
        return (
            f"{str(specifiers)!r} has no single '>=' lower bound "
            f"(expected >={expected})"
        )
    if found != expected:
        return f"lower bound is {found!r}, expected {expected!r}"
    return None


def _iter_project_requirements(
    data: dict[str, Any], *, source: str = ""
) -> list[tuple[str, str]]:
    """Return (location, requirement_string) pairs from project deps."""

    def loc(part: str) -> str:
        return f"{source}:{part}" if source else part

    project = data.get("project") or {}
    if not isinstance(project, dict):
        label = source or "root pyproject.toml"
        raise SystemExit(f"{label}: project table missing or invalid")
    entries: list[tuple[str, str]] = []
    for req in project.get("dependencies") or []:
        if isinstance(req, str):
            entries.append((loc("project.dependencies"), req))
    optional = project.get("optional-dependencies") or {}
    if not isinstance(optional, dict):
        label = source or "root pyproject.toml"
        raise SystemExit(f"{label}: optional-dependencies invalid")
    for extra, reqs in optional.items():
        for req in reqs or []:
            if isinstance(req, str):
                entries.append((loc(f"project.optional-dependencies.{extra}"), req))
    return entries


def _iter_root_requirements(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (location, requirement_string) pairs from root project deps."""
    return _iter_project_requirements(data)


def _pyproject_paths(repo_root: Path) -> list[Path]:
    paths = [repo_root / "pyproject.toml"]
    libs = repo_root / "libs"
    if libs.is_dir():
        paths.extend(sorted(libs.glob("*/pyproject.toml")))
    return [path for path in paths if path.is_file()]


def collect_mismatches(data: dict[str, Any], members: dict[str, str]) -> list[str]:
    """Return human-readable mismatch lines for root pins vs member versions."""
    mismatches: list[str] = []

    for location, req_str in _iter_root_requirements(data):
        try:
            req = Requirement(req_str)
        except InvalidRequirement as exc:
            mismatches.append(f"{location}: invalid requirement {req_str!r}: {exc}")
            continue

        expected = members.get(_canonical_name(req.name))
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

    return mismatches


def collect_package_pin_mismatches(
    repo_root: Path, package: str, version: str
) -> list[str]:
    """Return mismatch lines for ``package`` pins under ``repo_root``.

    Parameters
    ----------
    repo_root : Path
        Monorepo root containing ``pyproject.toml`` and ``libs/``.
    package : str
        Distribution name whose versioned requirements are audited.
    version : str
        Required ``>=`` lower bound (PEP 440).

    Returns
    -------
    list[str]
        Human-readable mismatches. Empty when every versioned requirement
        naming ``package`` has a single ``>=`` bound equal to ``version``,
        or when no versioned requirement names ``package``.
    """
    try:
        Version(version)
    except InvalidVersion as exc:
        raise SystemExit(f"invalid version {version!r}: {exc}") from exc

    target = _canonical_name(package)
    mismatches: list[str] = []

    for path in _pyproject_paths(repo_root):
        source = path.relative_to(repo_root).as_posix()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for location, req_str in _iter_project_requirements(data, source=source):
            try:
                req = Requirement(req_str)
            except InvalidRequirement as exc:
                mismatches.append(f"{location}: invalid requirement {req_str!r}: {exc}")
                continue

            if _canonical_name(req.name) != target:
                continue
            if not req.specifier:
                continue

            reason = _package_pin_reason(req.specifier, version)
            if reason is None:
                continue
            mismatches.append(f"{location}: {req.name} {reason}")

    return mismatches


def _check_root_member_pins() -> int:
    data = tomllib.loads(ROOT_PYPROJECT.read_text(encoding="utf-8"))
    members = _member_versions()
    mismatches = collect_mismatches(data, members)

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


def _check_package_pins(package: str, version: str) -> int:
    mismatches = collect_package_pin_mismatches(REPO_ROOT, package, version)
    if mismatches:
        print(
            f"Workspace pins for {package} drift from {version}:",
            file=sys.stderr,
        )
        for line in mismatches:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(f"OK: versioned {package} pins match >={version} (or none found).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the root-member check, or PACKAGE VERSION across root and libs."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2:
        return _check_package_pins(args[0], args[1])
    if args:
        print("usage: check_extra_pins.py [PACKAGE VERSION]", file=sys.stderr)
        return 2
    return _check_root_member_pins()


if __name__ == "__main__":
    raise SystemExit(main())
