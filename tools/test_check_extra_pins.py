"""Unit tests for extra/dependency pin alignment."""

import tomllib
from pathlib import Path

import pytest

from tools.check_extra_pins import collect_mismatches, collect_package_pin_mismatches

ROOT = Path(__file__).resolve().parents[1]


def test_matching_lower_bound_is_ok() -> None:
    data = {
        "project": {
            "dependencies": ["giskard-checks>=1.0.2b6,<2"],
            "optional-dependencies": {
                "scan": ["giskard-scan>=1.0.0b4,<2"],
                "full": ["giskard[scan]"],
            },
        }
    }
    members = {
        "giskard-checks": "1.0.2b6",
        "giskard-scan": "1.0.0b4",
    }
    assert collect_mismatches(data, members) == []


@pytest.mark.parametrize(
    ("req", "needle"),
    [
        ("giskard-scan>=1.0.0b2,<2", "1.0.0b2"),
        ("giskard-scan==1.0.0b4", "no single '>=' lower bound"),
        ("giskard_scan>=1.0.0b2,<2", "1.0.0b2"),
    ],
    ids=["drifted_lower_bound", "missing_ge", "pep503_alias"],
)
def test_mismatch_cases(req: str, needle: str) -> None:
    data = {"project": {"optional-dependencies": {"scan": [req]}}}
    members = {"giskard-scan": "1.0.0b4"}
    mismatches = collect_mismatches(data, members)
    assert len(mismatches) == 1
    assert needle in mismatches[0]


def _write_stale_scan_tree(tmp_path: Path, scan_req: str) -> None:
    """Root pins match 1.0.3; giskard-scan is the only drifted requirement."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["giskard-checks>=1.0.3,<2"]\n',
        encoding="utf-8",
    )
    scan_dir = tmp_path / "libs" / "giskard-scan"
    scan_dir.mkdir(parents=True)
    (scan_dir / "pyproject.toml").write_text(
        f'[project]\ndependencies = ["{scan_req}"]\n',
        encoding="utf-8",
    )


def test_stale_lib_pin_is_ignored_by_root_only_check(tmp_path: Path) -> None:
    _write_stale_scan_tree(tmp_path, "giskard-checks>=1.0.2,<2")
    root_data = tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))
    assert collect_mismatches(root_data, {"giskard-checks": "1.0.3"}) == []


def test_stale_giskard_scan_pin_is_a_package_mismatch(tmp_path: Path) -> None:
    _write_stale_scan_tree(tmp_path, "giskard-checks>=1.0.2,<2")
    mismatches = collect_package_pin_mismatches(tmp_path, "giskard-checks", "1.0.3")
    assert len(mismatches) == 1
    assert "libs/giskard-scan/pyproject.toml" in mismatches[0]
    assert "1.0.2" in mismatches[0]
    assert "1.0.3" in mismatches[0]


@pytest.mark.parametrize(
    "scan_req",
    [
        "giskard-checks==1.0.3",
        "giskard-checks~=1.0.3",
        "giskard-checks>1.0.2,<2",
    ],
    ids=["eq", "compatible_release", "gt"],
)
def test_package_pins_reject_non_ge_operators(tmp_path: Path, scan_req: str) -> None:
    _write_stale_scan_tree(tmp_path, scan_req)
    mismatches = collect_package_pin_mismatches(tmp_path, "giskard-checks", "1.0.3")
    assert len(mismatches) == 1
    assert "libs/giskard-scan/pyproject.toml" in mismatches[0]


def test_package_pins_ok_when_root_and_lib_match(tmp_path: Path) -> None:
    _write_stale_scan_tree(tmp_path, "giskard-checks[regorus]>=1.0.3,<2")
    assert collect_package_pin_mismatches(tmp_path, "giskard-checks", "1.0.3") == []


def test_package_pins_ok_when_no_requirement_names_package(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\noptional-dependencies = { full = ["giskard[all-checks]"] }\n',
        encoding="utf-8",
    )
    (tmp_path / "libs" / "giskard-checks").mkdir(parents=True)
    (tmp_path / "libs" / "giskard-checks" / "pyproject.toml").write_text(
        '[project]\noptional-dependencies = { all = ["giskard-checks[readability]"] }\n',
        encoding="utf-8",
    )
    assert collect_package_pin_mismatches(tmp_path, "giskard", "3.0.0") == []


def test_current_tree_giskard_has_no_versioned_pins() -> None:
    assert collect_package_pin_mismatches(ROOT, "giskard", "3.0.0") == []
