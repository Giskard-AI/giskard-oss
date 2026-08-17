"""Fail when a public API symbol is missing a docstring, or a docstring is not NumPy-style.

Ruff's D1xx rules cannot tell a public API from an internal helper, so they are
disabled repo-wide (see ``ruff.toml``) and presence is enforced here instead:
every name a package re-exports in ``__all__``, plus every public method and
property of an exported class, must carry a docstring. Ruff still enforces the
NumPy section style of whatever docstrings exist.

Ruff does not flag a Google-style ``Args:`` block — the name is not a numpydoc
section, so its D4xx rules never fire on it — hence the second scan here.

Run with ``make check-docstrings`` (part of ``make check``).
"""

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

PACKAGES = (
    "giskard.core",
    "giskard.llm",
    "giskard.agents",
    "giskard.checks",
    "giskard.scan",
)


SRC_ROOTS = tuple(Path(__file__).resolve().parent.parent.glob("libs/*/src"))

# Google/Sphinx section headers. NumPy writes these with a dashed underline
# instead, so a trailing colon means the docstring mixes conventions.
_GOOGLE_SECTION = re.compile(
    r"^[ \t]*(Args|Arguments|Attributes|Keyword Args|Note|Notes|Raises|Returns|Yields|Example|Examples)[ \t]*:[ \t]*$",
    re.MULTILINE,
)


def _google_sections(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            continue
        for match in _GOOGLE_SECTION.finditer(docstring):
            yield match.group(1)


def _has_docstring(obj: object) -> bool:
    return bool((getattr(obj, "__doc__", None) or "").strip())


def _underlying(member: object) -> object | None:
    """Return the function a class member wraps, or None if it is not documentable."""
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    if inspect.isfunction(member):
        return member
    return None


def _missing_members(cls: type, exported_name: str):
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if _underlying(member) is None:
            continue
        # inspect.getdoc walks the MRO, so an override inherits the base
        # method's docstring and only needs its own when it adds something.
        if not (inspect.getdoc(getattr(cls, name)) or "").strip():
            yield f"{exported_name}.{name}"


def _missing_for_package(package: str):
    module = importlib.import_module(package)
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name)
        if not (
            inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismodule(obj)
        ):
            continue  # re-exported constants carry no docstring of their own
        if not _has_docstring(obj):
            yield name
        if inspect.isclass(obj):
            yield from _missing_members(obj, name)


def main() -> int:
    """Report every undocumented public symbol; exit non-zero if any were found."""
    total = 0
    for package in PACKAGES:
        missing = sorted(_missing_for_package(package))
        total += len(missing)
        for name in missing:
            print(f"{package}.{name}: missing docstring")

    for root in SRC_ROOTS:
        for path in sorted(root.rglob("*.py")):
            for section in _google_sections(path):
                total += 1
                print(f"{path}: Google-style `{section}:` section")

    if total:
        print(
            f"\n{total} docstring problem(s).\n"
            "Public APIs use NumPy-style docstrings; see the Docstrings section of CONTRIBUTING.md.",
            file=sys.stderr,
        )
        return 1

    print("All public API symbols documented, no Google-style sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
