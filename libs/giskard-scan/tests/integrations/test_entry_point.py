from unittest.mock import AsyncMock, MagicMock

import pytest
from giskard.checks import SuiteResult


async def test_third_party_scan_dispatches_to_adapter(monkeypatch):
    import giskard.scan.integrations._registry as reg

    expected = SuiteResult(results=[], duration_ms=42)
    adapter_instance = MagicMock()
    adapter_instance.run = AsyncMock(return_value=expected)
    adapter_class = MagicMock(return_value=adapter_instance)

    monkeypatch.setitem(reg._REGISTRY, "_test_tool", adapter_class)

    from giskard.scan import third_party_scan

    result = await third_party_scan(target=lambda p: "ok", tool="_test_tool", foo="bar")

    assert result is expected
    adapter_class.assert_called_once_with()
    adapter_instance.run.assert_called_once()
    _, call_kwargs = adapter_instance.run.call_args
    assert call_kwargs.get("foo") == "bar"


async def test_third_party_scan_unknown_tool_raises():
    from giskard.scan import third_party_scan

    with pytest.raises(ValueError, match="Unknown tool"):
        await third_party_scan(target=lambda p: "ok", tool="_no_such_tool_xyz")
