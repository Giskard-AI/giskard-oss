"""Tests for Hub format export."""

from giskard.checks.core.result import SuiteResult
from giskard.checks.export import hub as hub_module
from giskard.checks.export.hub import to_hub_format


def test_to_hub_format_pass_rate_null_when_empty() -> None:
    """Empty suites serialize pass_rate as JSON null for the Hub wire payload.

    Hub Metric.success_rate is already float | None; SuiteResult.pass_rate follows
    the same zero-denominator convention and ships through model_dump.
    """
    payload = to_hub_format(SuiteResult(results=[], duration_ms=0))
    assert "pass_rate" in payload
    assert payload["pass_rate"] is None


def test_to_hub_format_emits_aggregate_paid_intent_telemetry(monkeypatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    tags: list[tuple[str, str]] = []
    call_order: list[str] = []
    original_model_dump = SuiteResult.model_dump

    def model_dump_spy(self, *args, **kwargs):
        call_order.append("model_dump")
        return original_model_dump(self, *args, **kwargs)

    monkeypatch.setattr(SuiteResult, "model_dump", model_dump_spy)
    monkeypatch.setattr(
        hub_module,
        "telemetry_capture",
        lambda event, *, properties: (
            call_order.append("telemetry_capture"),
            events.append((event, properties)),
        ),
    )
    monkeypatch.setattr(
        hub_module,
        "telemetry_tag",
        lambda key, value: tags.append((key, value)),
    )

    result = SuiteResult(
        results=[], duration_ms=123, recommendation="private recommendation"
    )
    to_hub_format(result)

    assert call_order == ["model_dump", "telemetry_capture"]
    assert tags == [
        ("giskard_component", "export"),
        ("giskard_operation", "to_hub_format"),
    ]
    assert events == [
        (
            "checks_hub_exported",
            {
                "integration": "giskard-checks",
                "scenario_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "errored_count": 0,
                "skipped_count": 0,
                "has_recommendation": True,
            },
        )
    ]
    assert "private recommendation" not in repr(events)
