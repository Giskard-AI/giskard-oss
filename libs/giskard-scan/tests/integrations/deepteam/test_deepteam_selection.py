"""Tests for deepteam vulnerability/attack name resolution."""

import pytest

pytest.importorskip("deepteam")

from giskard.scan.integrations.deepteam import _selection


def test_defaults_resolve_to_instances():
    vulns = _selection.resolve_vulnerabilities(None)
    attacks = _selection.resolve_attacks(None, singleturn=False)
    assert len(vulns) == len(_selection.DEFAULT_VULNERABILITIES)
    assert len(attacks) == len(_selection.DEFAULT_ATTACKS)
    # Instances, not classes.
    assert all(not isinstance(v, type) for v in vulns)
    assert all(not isinstance(a, type) for a in attacks)


def test_explicit_names_resolve():
    vulns = _selection.resolve_vulnerabilities(["Bias"])
    assert len(vulns) == 1
    assert type(vulns[0]).__name__ == "Bias"


def test_unknown_vulnerability_raises_listing_valid_names():
    with pytest.raises(ValueError, match="Bias"):
        _selection.resolve_vulnerabilities(["NotAThing"])


def test_unknown_attack_raises():
    with pytest.raises(ValueError, match="PromptInjection"):
        _selection.resolve_attacks(["NotAThing"], singleturn=False)


def test_singleturn_drops_multiturn_attacks():
    names = ["PromptInjection", "LinearJailbreaking"]
    multiturn = _selection.resolve_attacks(names, singleturn=False)
    singleturn = _selection.resolve_attacks(names, singleturn=True)
    assert {type(a).__name__ for a in multiturn} == {
        "PromptInjection",
        "LinearJailbreaking",
    }
    # LinearJailbreaking is multi-turn -> dropped.
    assert {type(a).__name__ for a in singleturn} == {"PromptInjection"}


def test_empty_list_resolves_to_empty():
    assert _selection.resolve_vulnerabilities([]) == []
    assert _selection.resolve_attacks([], singleturn=False) == []
