"""Rewrite >= lower bounds for a workspace package across pyproject files.

Used by the release workflow after ``uv version`` so root extras (and
downstream lib pins) stay aligned with the member version that
``check_extra_pins PACKAGE VERSION`` enforces.
"""

import re
import sys
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]

# PEP 508 name + optional extras + >= lower bound. Trailing upper bounds,
# markers, and comments are left untouched.
_PIN_RE = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?P<extra>\[[^\]]+\])?"
    r"(?P<op>\s*>=\s*)"
    r"(?P<version>[^,<\"'\s]+)"
)


def rewrite_text(text: str, package: str, new_version: str) -> tuple[str, int]:
    """Replace ``>=`` pins for ``package`` in a pyproject document.

    Parameters
    ----------
    text : str
        Raw ``pyproject.toml`` contents.
    package : str
        Distribution name to update (hyphen or underscore form).
    new_version : str
        Replacement lower bound (PEP 440).

    Returns
    -------
    tuple[str, int]
        Rewritten text and the number of pins that named ``package``.
    """
    target = canonicalize_name(package)
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        if canonicalize_name(match.group("name")) != target:
            return match.group(0)
        count += 1
        extra = match.group("extra") or ""
        return f"{match.group('name')}{extra}{match.group('op')}{new_version}"

    return _PIN_RE.sub(repl, text), count


def pyproject_paths(repo_root: Path) -> list[Path]:
    """Return root and ``libs/*/pyproject.toml`` paths that exist."""
    paths = [repo_root / "pyproject.toml"]
    libs = repo_root / "libs"
    if libs.is_dir():
        paths.extend(sorted(libs.glob("*/pyproject.toml")))
    return [path for path in paths if path.is_file()]


def bump_pins(
    repo_root: Path, package: str, new_version: str
) -> list[tuple[Path, int]]:
    """Write updated pins for ``package`` under ``repo_root``.

    Parameters
    ----------
    repo_root : Path
        Monorepo root containing ``pyproject.toml`` and ``libs/``.
    package : str
        Distribution name whose ``>=`` pins should move.
    new_version : str
        Replacement lower bound (PEP 440).

    Returns
    -------
    list[tuple[Path, int]]
        Files that changed, with the pin count in each file.
    """
    try:
        Version(new_version)
    except InvalidVersion as exc:
        raise SystemExit(f"invalid version {new_version!r}: {exc}") from exc

    results: list[tuple[Path, int]] = []
    for path in pyproject_paths(repo_root):
        original = path.read_text(encoding="utf-8")
        updated, count = rewrite_text(original, package, new_version)
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        results.append((path, count))
    return results


def main(argv: list[str] | None = None) -> int:
    """Rewrite pins for ``PACKAGE VERSION``. Returns a process exit code."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: bump_workspace_pins.py PACKAGE VERSION", file=sys.stderr)
        return 2

    package, version = args
    results = bump_pins(REPO_ROOT, package, version)
    if not results:
        print(f"No {package} pins to update.")
        return 0
    for path, count in results:
        print(f"Updated {count} pin(s) in {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
