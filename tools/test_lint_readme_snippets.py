from pathlib import Path

from tools.lint_readme_snippets import lint_markdown


def test_lint_markdown_checks_spaced_python_fences(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        '``` Python \nCheckResult.success("bad")\n```\n',
        encoding="utf-8",
    )

    assert lint_markdown(readme) == [
        f"{readme}:1: positional CheckResult.success(str) is forbidden; use message="
    ]


def test_lint_markdown_checks_positional_f_strings(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        '```python\nCheckResult.failure(f"bad {value}")\n```\n',
        encoding="utf-8",
    )

    assert lint_markdown(readme) == [
        f"{readme}:1: positional CheckResult.failure(str) is forbidden; use message="
    ]
