"""Tests for shared list_scan_items discovery."""

import pytest
from giskard.scan import list_scan_items


def test_list_scan_items_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        # Deliberately off-Literal: the runtime guard must still hold for
        # callers that do not type-check.
        list_scan_items("not-a-tool")  # pyright: ignore[reportArgumentType]


def test_list_scan_items_garak() -> None:
    # Unit CI may see a top-level ``garak`` without the real plugin package.
    pytest.importorskip("garak._plugins")
    from giskard.scan.integrations.garak import list_probes

    assert list_scan_items("garak") == list_probes()
    assert "probes.goodside.ThreatenJSON" in list_scan_items("garak")


def test_list_scan_items_deepteam() -> None:
    pytest.importorskip("deepteam.vulnerabilities")
    from giskard.scan.integrations.deepteam import list_attacks, list_vulnerabilities

    names = list_scan_items("deepteam")
    assert "Bias" in names
    assert "PromptInjection" in names
    assert set(names) == {*list_vulnerabilities(), *list_attacks()}
