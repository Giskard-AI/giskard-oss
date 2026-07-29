"""Tests for shared list_scan_items discovery."""

import pytest


def test_list_scan_items_rejects_unknown_tool() -> None:
    from giskard.scan import list_scan_items

    with pytest.raises(ValueError, match="Unknown tool"):
        list_scan_items("not-a-tool")


def test_list_scan_items_garak() -> None:
    pytest.importorskip("garak")
    from giskard.scan import list_scan_items
    from giskard.scan.integrations.garak import list_probes

    assert list_scan_items("garak") == list_probes()
    assert "probes.goodside.ThreatenJSON" in list_scan_items("garak")


def test_list_scan_items_deepteam() -> None:
    pytest.importorskip("deepteam")
    from giskard.scan import list_scan_items
    from giskard.scan.integrations.deepteam import list_attacks, list_vulnerabilities

    names = list_scan_items("deepteam")
    assert "Bias" in names
    assert "PromptInjection" in names
    assert set(names) == {*list_vulnerabilities(), *list_attacks()}
