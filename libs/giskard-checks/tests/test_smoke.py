"""Smoke tests for import safety without optional dependencies.

- The first two tests run on every install: they verify the package imports
  safely and exposes its public API without any optional dependency.
- The last test patches ``regorus`` import failure and verifies that
  ``RegoPolicy`` fails fast at instantiation with a helpful validation error.
"""

import importlib

import pytest
from giskard.checks.utils import optional_deps
from pydantic import ValidationError


def test_package_import_does_not_raise():
    import giskard.checks  # noqa: F401


def test_public_api_is_accessible():
    import giskard.checks as m

    for name in [
        "Check",
        "CheckResult",
        "CheckStatus",
        "RegoPolicy",
        "Interaction",
        "Trace",
        "Target",
    ]:
        assert hasattr(m, name), f"giskard.checks missing attribute: {name}"


def test_rego_policy_raises_validation_error_when_regorus_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from giskard.checks import RegoPolicy

    real_import_module = importlib.import_module

    def fake_import_module(name: str, /, *args, **kwargs):
        if name == "regorus":
            raise ImportError("missing regorus")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(optional_deps.importlib, "import_module", fake_import_module)

    with pytest.raises(ValidationError, match="giskard-checks\\[regorus\\]"):
        _ = RegoPolicy(
            policy="package giskard\nallow := true", rule="data.giskard.allow"
        )
