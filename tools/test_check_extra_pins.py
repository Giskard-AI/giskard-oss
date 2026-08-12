"""Unit tests for root extra/dependency pin alignment."""

import pytest

from tools.check_extra_pins import collect_mismatches


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
