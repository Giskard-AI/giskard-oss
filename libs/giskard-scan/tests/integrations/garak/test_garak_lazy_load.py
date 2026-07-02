"""Smoke tests for garak integration import safety.

- Import tests patch garak as unavailable and verify integration modules
  import without raising (garak is loaded lazily).
- The run test verifies GarakScanAdapter fails fast with a helpful ImportError
  when garak is absent.

The ``block_garak_import`` fixture lives in conftest.py; see it for how the
sys.modules purge works and why later tests must patch via the live module.
"""

import pytest


def test_garak_available_returns_bool() -> None:
    from giskard.scan.integrations.garak._adapter import garak_available

    assert isinstance(garak_available(), bool)


def test_garak_adapter_module_import_does_not_raise(
    block_garak_import: None,
) -> None:
    from giskard.scan.integrations.garak import GarakScanAdapter

    assert GarakScanAdapter is not None


def test_third_party_scan_import_does_not_raise(block_garak_import: None) -> None:
    from giskard.scan.integrations import third_party_scan
    from giskard.scan.third_party import third_party_scan as compat_third_party_scan

    assert callable(third_party_scan)
    assert compat_third_party_scan is third_party_scan


async def test_garak_adapter_run_raises_import_error_when_garak_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from giskard.scan.integrations.garak import GarakScanAdapter

    monkeypatch.setattr(
        "giskard.scan.integrations.garak._adapter.garak_available",
        lambda: False,
    )

    with pytest.raises(ImportError, match="giskard-scan\\[garak\\]"):
        await GarakScanAdapter().run(target=lambda p: "ok")
