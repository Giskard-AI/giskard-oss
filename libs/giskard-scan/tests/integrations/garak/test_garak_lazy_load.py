"""Smoke tests for garak integration import safety.

- Import tests patch garak as unavailable and verify integration modules
  import without raising (garak is loaded lazily).
- The run test verifies GarakScanAdapter fails fast with a helpful ImportError
  when garak is absent.

- The collection-gate tests pin the contract in ``libs/giskard-scan/conftest.py``,
  which keeps the optional-extra ``src`` trees out of ``--doctest-modules`` collection
  when the extra is absent. Those trees are skipped rather than fixed, so without these
  tests a broken gate would fail loudly only in whichever CI job happens to run from the
  wrong CWD.

The ``block_garak_import`` fixture lives in conftest.py; see it for how the
sys.modules purge works and why later tests must patch via the live module.
"""

import importlib.util
from pathlib import Path

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

    assert callable(third_party_scan)


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


def _lib_conftest_namespace() -> tuple[tuple[str, ...], list[str]]:
    """Execute the lib-root conftest in isolation, returning (extras, ignore globs).

    Loaded by path rather than imported: pytest has already imported the real one as a
    plugin, and importing a second copy under a different name would shadow it.
    """
    conftest_path = Path(__file__).parents[3] / "conftest.py"
    namespace: dict[str, object] = {"__file__": str(conftest_path)}
    exec(compile(conftest_path.read_text(), str(conftest_path), "exec"), namespace)

    extras = namespace["_OPTIONAL_EXTRAS"]
    ignored = namespace["collect_ignore_glob"]
    assert isinstance(extras, tuple)
    assert isinstance(ignored, list)
    return tuple(str(extra) for extra in extras), [str(pattern) for pattern in ignored]


def test_lib_conftest_ignores_src_tree_of_every_absent_extra() -> None:
    """Absent extras must have their src tree excluded from collection.

    This is what stops ``--doctest-modules`` from importing modules whose optional
    dependency is missing, regardless of the CWD pytest was invoked from.
    """
    extras, ignored = _lib_conftest_namespace()

    for extra in extras:
        pattern = f"src/giskard/scan/integrations/{extra}/*"
        if importlib.util.find_spec(extra) is None:
            assert pattern in ignored, (
                f"{extra} is not installed but its src tree is still collected; "
                "--doctest-modules will import it and fail"
            )
        else:
            assert pattern not in ignored, (
                f"{extra} is installed but its src tree is excluded; "
                f"make test-{extra} would silently collect nothing"
            )


def test_lib_conftest_extras_match_integration_directories() -> None:
    """The gate assumes each extra's import name equals its directory name."""
    extras = set(_lib_conftest_namespace()[0])

    integrations_dir = Path(__file__).parents[3] / "src/giskard/scan/integrations"
    on_disk = {
        path.name
        for path in integrations_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }

    assert extras <= on_disk, (
        f"conftest gates extras with no integration directory: {sorted(extras - on_disk)}"
    )


def test_garak_package_imports_without_garak_installed(
    block_garak_import: None,
) -> None:
    """The public surface must import with garak absent -- no importorskip.

    Pins the lazy boundary the conftest gate relies on: ``__init__`` pulls only from
    ``_adapter.py``, which imports ``TargetGenerator`` inside a function body.
    """
    from giskard.scan.integrations.garak import GarakScanAdapter, garak_available

    assert GarakScanAdapter is not None
    assert garak_available() is False
