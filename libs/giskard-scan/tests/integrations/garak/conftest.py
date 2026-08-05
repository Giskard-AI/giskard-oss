"""Shared fixtures for the garak integration tests.

The one piece of shared machinery is ``block_garak_import``: it fakes garak as
absent and purges the integration modules so the lazy-import path can be exercised.
Because it deletes those modules from ``sys.modules`` (on both setup and teardown),
any later test that captured a module reference at import time would bind a *stale*
object. That is why the resolver-patching helpers in the other test files
(``_patch_resolvers``, ``_patch_probe``) patch and return the adapter from the
*live* module rather than a top-level import.
"""

import importlib
import importlib.util
import sys

import pytest

# Every module the garak integration lazily imports; purged so the lazy-import
# path runs under the faked-absent garak.
_INTEGRATION_MODULES = (
    "giskard.scan.integrations.garak._generator",
    "giskard.scan.integrations.garak._adapter",
    "giskard.scan.integrations.garak",
    "giskard.scan.integrations._entry_point",
    "giskard.scan.integrations",
)


def _purge_integration_modules() -> None:
    for module_name in list(sys.modules):
        if module_name in _INTEGRATION_MODULES or module_name.startswith(
            "giskard.scan.integrations.garak."
        ):
            del sys.modules[module_name]


@pytest.fixture
def block_garak_import(monkeypatch: pytest.MonkeyPatch):
    """Make garak look uninstalled and force a fresh integration import.

    Purges the integration modules on setup (so the test body re-imports them
    under the fakes) and again on teardown (so the fake-built objects don't leak
    into later tests — the next importer gets a clean, real-garak module).
    """
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def fake_find_spec(name, package=None):
        if name == "garak" or (isinstance(name, str) and name.startswith("garak.")):
            return None
        return real_find_spec(name, package)

    def fake_import_module(name: str, /, *args, **kwargs):
        if name == "garak" or name.startswith("garak."):
            raise ImportError("missing garak")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    _purge_integration_modules()
    yield
    _purge_integration_modules()
