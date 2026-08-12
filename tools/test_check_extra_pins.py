"""Unit tests for root extra/dependency pin alignment."""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "check_extra_pins.py"
_SPEC = importlib.util.spec_from_file_location("check_extra_pins", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
collect_mismatches = _MODULE.collect_mismatches


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


def test_drifted_lower_bound_is_reported() -> None:
    data = {
        "project": {
            "optional-dependencies": {
                "scan": ["giskard-scan>=1.0.0b2,<2"],
            },
        }
    }
    members = {"giskard-scan": "1.0.0b4"}
    mismatches = collect_mismatches(data, members)
    assert len(mismatches) == 1
    assert "1.0.0b2" in mismatches[0]
    assert "1.0.0b4" in mismatches[0]


def test_missing_ge_lower_bound_is_reported() -> None:
    data = {
        "project": {
            "optional-dependencies": {
                "scan": ["giskard-scan==1.0.0b4"],
            },
        }
    }
    members = {"giskard-scan": "1.0.0b4"}
    mismatches = collect_mismatches(data, members)
    assert len(mismatches) == 1
    assert "no single '>=' lower bound" in mismatches[0]


def test_pep503_alias_pin_still_checked() -> None:
    data = {
        "project": {
            "optional-dependencies": {
                "scan": ["giskard_scan>=1.0.0b2,<2"],
            },
        }
    }
    members = {"giskard-scan": "1.0.0b4"}
    mismatches = collect_mismatches(data, members)
    assert len(mismatches) == 1
    assert "1.0.0b2" in mismatches[0]
