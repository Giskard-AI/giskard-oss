"""Unit tests for rewriting workspace-member >= pins."""

import tomllib
from pathlib import Path

from tools.bump_workspace_pins import bump_pins, rewrite_text
from tools.check_extra_pins import _member_versions, collect_mismatches

ROOT = Path(__file__).resolve().parents[1]


def test_rewrite_preserves_extras_comments_and_other_packages() -> None:
    text = (
        'dependencies = ["giskard-checks>=1.0.2,<2"]\n'
        'regorus = ["giskard-checks[regorus]>=1.0.2,<2"]  # optional\n'
        'litellm = ["giskard-agents[litellm]>=1.0.2,<2"]\n'
        'requires = ["hatchling>=1.25.0"]\n'
    )
    updated, count = rewrite_text(text, "giskard-checks", "1.0.3")
    assert count == 2
    assert 'dependencies = ["giskard-checks>=1.0.3,<2"]' in updated
    assert 'regorus = ["giskard-checks[regorus]>=1.0.3,<2"]  # optional' in updated
    assert 'litellm = ["giskard-agents[litellm]>=1.0.2,<2"]' in updated
    assert 'requires = ["hatchling>=1.25.0"]' in updated


def test_rewrite_does_not_match_package_name_prefix() -> None:
    text = 'all-checks = ["giskard-checks[all]>=1.0.2,<2"]\nfull = ["giskard[all-checks]>=3.0.0,<4"]\n'
    updated, count = rewrite_text(text, "giskard", "3.1.0")
    assert count == 1
    assert "giskard-checks[all]>=1.0.2,<2" in updated
    assert "giskard[all-checks]>=3.1.0,<4" in updated


def test_rewrite_accepts_pep503_alias() -> None:
    text = 'scan = ["giskard_scan>=1.0.0,<2"]\n'
    updated, count = rewrite_text(text, "giskard-scan", "1.0.1")
    assert count == 1
    assert updated == 'scan = ["giskard_scan>=1.0.1,<2"]\n'


def test_bump_pins_updates_root_and_downstream_libs(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    scan_dir = tmp_path / "libs" / "giskard-scan"
    scan_dir.mkdir(parents=True)
    root_pyproject.write_text(
        '[project]\ndependencies = ["giskard-checks>=1.0.2,<2"]\n',
        encoding="utf-8",
    )
    (scan_dir / "pyproject.toml").write_text(
        '[project]\ndependencies = ["giskard-checks>=1.0.2,<2", "giskard-agents>=1.0.2,<2"]\n',
        encoding="utf-8",
    )
    (tmp_path / "libs" / "giskard-core").mkdir()
    (tmp_path / "libs" / "giskard-core" / "pyproject.toml").write_text(
        '[project]\nname = "giskard-core"\nversion = "1.0.1"\n',
        encoding="utf-8",
    )

    results = bump_pins(tmp_path, "giskard-checks", "1.0.3")
    changed = {path.relative_to(tmp_path).as_posix(): n for path, n in results}
    assert changed == {
        "pyproject.toml": 1,
        "libs/giskard-scan/pyproject.toml": 1,
    }
    assert "giskard-checks>=1.0.3,<2" in root_pyproject.read_text(encoding="utf-8")
    scan_text = (scan_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert "giskard-checks>=1.0.3,<2" in scan_text
    assert "giskard-agents>=1.0.2,<2" in scan_text


def test_simulated_checks_release_keeps_root_pins_aligned() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    members = _member_versions()
    new_version = "9.9.9"
    rewritten, count = rewrite_text(text, "giskard-checks", new_version)
    assert count >= 1
    members["giskard-checks"] = new_version
    assert collect_mismatches(tomllib.loads(rewritten), members) == []
