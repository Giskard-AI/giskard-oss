"""Hub checks (``hub_*`` kinds) load as skipped placeholders when unavailable.

Hub checks are registered by the Hub runtime. Outside of it, loading a suite
that references one must warn and skip that check instead of failing to load,
while unregistered non-Hub kinds (e.g. a custom check whose module was not
imported) keep raising.
"""

import warnings
from typing import Any

import pytest
from giskard.checks import (
    Check,
    CheckResult,
    Scenario,
    StringMatching,
    Trace,
    UnavailableCheckWarning,
    UnavailableHubCheck,
)
from pydantic import TypeAdapter, ValidationError

CHECK_ADAPTER: TypeAdapter[Check[Any, Any, Any]] = TypeAdapter(Check)

HUB_PAYLOAD = {
    "kind": "hub_correctness",
    "name": "correctness",
    "description": "Answer matches the reference",
    "reference": "Paris",
    "text_key": "trace.last.outputs",
}


@Check.register("hub_available_in_env")
class AvailableHubCheck(Check[Any, Any, Trace[Any, Any]]):
    """Stands in for a Hub check whose module is imported (running on the Hub)."""

    async def run(self, trace: Trace[Any, Any]) -> CheckResult:
        return CheckResult.success()


def _load_hub_check() -> Check[Any, Any, Any]:
    with pytest.warns(UnavailableCheckWarning, match="'hub_correctness'"):
        return CHECK_ADAPTER.validate_python(HUB_PAYLOAD)


def test_unavailable_hub_check_loads_as_placeholder():
    check = _load_hub_check()

    assert isinstance(check, UnavailableHubCheck)
    assert check.kind == "hub_correctness"
    assert check.name == "correctness"
    assert check.description == "Answer matches the reference"


def test_unavailable_hub_check_preserves_original_payload():
    """Dumps and specs report the Hub kind and config, not placeholder fields."""
    check = _load_hub_check()

    assert check.model_dump() == HUB_PAYLOAD
    assert check.to_spec() == {
        "kind": "hub_correctness",
        "reference": "Paris",
        "text_key": "trace.last.outputs",
    }
    with pytest.warns(UnavailableCheckWarning):
        reloaded = CHECK_ADAPTER.validate_python(check.model_dump())
    assert reloaded == check


async def test_unavailable_hub_check_runs_as_skip():
    check = _load_hub_check()

    result = await check.run(Trace())

    assert result.skipped
    assert result.message is not None
    assert "hub_correctness" in result.message


async def test_scenario_with_unavailable_hub_check_runs_other_checks():
    with pytest.warns(UnavailableCheckWarning):
        scenario = Scenario.model_validate(
            {
                "name": "mixed",
                "steps": [
                    {
                        "interacts": [],
                        "checks": [
                            HUB_PAYLOAD,
                            StringMatching(keyword="hi", text="hi there").model_dump(),
                        ],
                    }
                ],
            }
        )

    result = await scenario.run()

    hub_result, string_result = result.steps[0].results
    assert hub_result.skipped
    assert hub_result.details["check_kind"] == "hub_correctness"
    assert string_result.passed


def test_registered_hub_check_is_used_without_warning():
    """When the environment registers a Hub kind, the real class wins."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnavailableCheckWarning)
        check = CHECK_ADAPTER.validate_python({"kind": "hub_available_in_env"})

    assert isinstance(check, AvailableHubCheck)


def test_unregistered_custom_check_still_raises():
    """Non-Hub unknown kinds are not mistaken for Hub checks."""
    with pytest.raises(ValidationError, match="Kind my_custom_check is not registered"):
        CHECK_ADAPTER.validate_python({"kind": "my_custom_check"})


def test_warning_can_be_escalated_to_error():
    """Environments expected to provide Hub checks can make this fail loudly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnavailableCheckWarning)
        with pytest.raises(UnavailableCheckWarning):
            CHECK_ADAPTER.validate_python(HUB_PAYLOAD)
