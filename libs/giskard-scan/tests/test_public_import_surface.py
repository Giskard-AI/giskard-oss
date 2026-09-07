import ast
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
UPSTREAM_ROOTS = ("giskard.agents", "giskard.checks")


def _iter_python_files(root: Path):
    yield from sorted(root.rglob("*.py"))


def _format_import_names(names: list[ast.alias]) -> str:
    parts: list[str] = []
    for alias in names:
        if alias.asname:
            parts.append(f"{alias.name} as {alias.asname}")
        else:
            parts.append(alias.name)
    return ", ".join(parts)


def _format_import_statement(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return f"from {module} import {_format_import_names(node.names)}"
    return f"import {_format_import_names(node.names)}"


def _violating_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(module.startswith(f"{root}.") for root in UPSTREAM_ROOTS):
                yield node.lineno, _format_import_statement(node)
        elif isinstance(node, ast.Import):
            violating_names = [
                alias
                for alias in node.names
                if any(alias.name.startswith(f"{root}.") for root in UPSTREAM_ROOTS)
            ]
            if violating_names:
                yield node.lineno, f"import {_format_import_names(violating_names)}"


def test_scan_src_imports_upstream_libs_only_from_package_roots():
    violations = [
        f"{path.relative_to(PACKAGE_SRC.parent)}:{line}: {statement}"
        for path in _iter_python_files(PACKAGE_SRC)
        for line, statement in _violating_imports(path)
    ]

    allowed_roots = ", ".join(f"`{root}`" for root in UPSTREAM_ROOTS)
    assert violations == [], (
        "Import upstream libs from package roots only, not submodules.\n"
        f"Allowed roots: {allowed_roots}\n"
        f"Forbidden: `from <root>.<submodule> import ...`\n"
        f"Violations ({len(violations)}):\n" + "\n".join(f"  - {v}" for v in violations)
    )
