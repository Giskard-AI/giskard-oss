"""Tests for deepteam vulnerability/attack name resolution."""

import pytest

pytest.importorskip("deepteam")

from giskard.scan.integrations.deepteam import _selection


def test_defaults_resolve_to_instances():
    vulns, skipped_v = _selection.resolve_vulnerabilities(None)
    attacks, skipped_a = _selection.resolve_attacks(None, singleturn=False)
    assert skipped_v == []
    assert skipped_a == []
    assert len(vulns) == len(_selection.DEFAULT_VULNERABILITIES)
    assert len(attacks) == len(_selection.DEFAULT_ATTACKS)
    assert all(not isinstance(v, type) for v in vulns)
    assert all(not isinstance(a, type) for a in attacks)


def test_explicit_names_resolve():
    vulns, skipped = _selection.resolve_vulnerabilities(["Bias"])
    assert skipped == []
    assert len(vulns) == 1
    assert type(vulns[0]).__name__ == "Bias"


def test_unknown_vulnerability_is_skipped():
    vulns, skipped = _selection.resolve_vulnerabilities(["NotAThing", "Bias"])
    assert len(vulns) == 1
    assert type(vulns[0]).__name__ == "Bias"
    assert len(skipped) == 1
    assert skipped[0].name == "NotAThing"
    assert skipped[0].reason == "unknown"


def test_unknown_attack_is_skipped():
    attacks, skipped = _selection.resolve_attacks(
        ["NotAThing", "PromptInjection"], singleturn=False
    )
    assert {type(a).__name__ for a in attacks} == {"PromptInjection"}
    assert len(skipped) == 1
    assert skipped[0].name == "NotAThing"
    assert skipped[0].reason == "unknown"


def test_singleturn_drops_multiturn_attacks():
    names = ["PromptInjection", "LinearJailbreaking"]
    multiturn, skipped_m = _selection.resolve_attacks(names, singleturn=False)
    singleturn, skipped_s = _selection.resolve_attacks(names, singleturn=True)
    assert skipped_m == []
    assert {type(a).__name__ for a in multiturn} == {
        "PromptInjection",
        "LinearJailbreaking",
    }
    assert {type(a).__name__ for a in singleturn} == {"PromptInjection"}
    assert len(skipped_s) == 1
    assert skipped_s[0].name == "LinearJailbreaking"
    assert "singleturn" in skipped_s[0].reason


def test_empty_list_resolves_to_empty():
    assert _selection.resolve_vulnerabilities([]) == ([], [])
    assert _selection.resolve_attacks([], singleturn=False) == ([], [])


def test_list_helpers():
    assert "Bias" in _selection.list_vulnerabilities()
    assert "PromptInjection" in _selection.list_attacks()
    assert "CrescendoJailbreaking" in _selection.list_attacks()
