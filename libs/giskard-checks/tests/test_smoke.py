"""Smoke tests for import safety without optional dependencies.

- The first two tests run on every install: they verify the package imports
  safely and exposes its public API without any optional dependency.
- The last test uses pytest.skipif to guard against regorus being present;
  it verifies that RegoPolicy.run() raises a helpful error when regorus is absent.
"""

import importlib.util

import pytest


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
    ]:
        assert hasattr(m, name), f"giskard.checks missing attribute: {name}"


@pytest.mark.skipif(
    importlib.util.find_spec("regorus") is not None,
    reason="regorus is installed; ImportError would not be raised",
)
async def test_rego_policy_raises_import_error_when_regorus_absent():
    from giskard.checks import CheckStatus, Interaction, RegoPolicy, Trace

    check = RegoPolicy(
        policy="package giskard\nallow := true", rule="data.giskard.allow"
    )
    trace = await Trace.from_interactions(Interaction(inputs="test", outputs={}))
    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert "giskard-checks[regorus]" in (result.message or "")
