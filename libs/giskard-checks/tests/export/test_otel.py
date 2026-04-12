import builtins
from types import TracebackType
from typing import Any

import pytest
from giskard.checks import (
    CheckResult,
    CheckStatus,
    Interaction,
    Metric,
    ScenarioResult,
    SuiteResult,
    Trace,
)
from giskard.checks import TestCaseResult as GiskardTestCaseResult
from giskard.checks.export.otel import OTelExporter


class FakeSpan:
    def __init__(self, name: str, attributes: dict[str, Any] | None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "FakeSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_as_current_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> FakeSpan:
        span = FakeSpan(name, attributes)
        self.spans.append(span)
        return span


class FakeInstrument:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records: list[tuple[str, float, dict[str, Any]]] = []

    def record(self, value: float, attributes: dict[str, Any]) -> None:
        self.records.append(("record", value, attributes))

    def add(self, value: float, attributes: dict[str, Any]) -> None:
        self.records.append(("add", value, attributes))


class FakeMeter:
    def __init__(self) -> None:
        self.instruments: dict[str, FakeInstrument] = {}

    def create_histogram(
        self,
        name: str,
        *,
        unit: str,
        description: str,
    ) -> FakeInstrument:
        return self._instrument(name)

    def create_counter(
        self,
        name: str,
        *,
        unit: str,
        description: str,
    ) -> FakeInstrument:
        return self._instrument(name)

    def _instrument(self, name: str) -> FakeInstrument:
        instrument = FakeInstrument(name)
        self.instruments[name] = instrument
        return instrument


class FakeProvider:
    def __init__(self, *, supports_timeout: bool = True) -> None:
        self.supports_timeout = supports_timeout
        self.flushed = False
        self.shutdown_called = False
        self.timeout_millis: int | None = None

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        if not self.supports_timeout and timeout_millis is not None:
            raise TypeError("unexpected keyword argument 'timeout_millis'")
        self.flushed = True
        self.timeout_millis = timeout_millis
        return True

    def shutdown(self, timeout_millis: int | None = None) -> None:
        if not self.supports_timeout and timeout_millis is not None:
            raise TypeError("unexpected keyword argument 'timeout_millis'")
        self.shutdown_called = True
        self.timeout_millis = timeout_millis


def _sample_suite_result() -> SuiteResult:
    return SuiteResult(
        results=[
            ScenarioResult(
                scenario_name="scenario_pass",
                steps=[
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="grounded",
                                metrics=[Metric(name="score", value=0.95)],
                                details={"check_name": "Groundedness"},
                            ),
                            CheckResult(
                                status=CheckStatus.PASS,
                                message="relevant",
                                details={"check_name": "AnswerRelevance"},
                            ),
                        ],
                        duration_ms=100,
                    )
                ],
                duration_ms=100,
                final_trace=Trace(
                    interactions=[
                        Interaction(
                            inputs={"question": "What is Giskard?"},
                            outputs={"answer": "An evaluation framework."},
                            metadata={"model": "test-model"},
                        )
                    ]
                ),
            ),
            ScenarioResult(
                scenario_name="scenario_fail",
                steps=[
                    GiskardTestCaseResult(
                        results=[
                            CheckResult(
                                status=CheckStatus.FAIL,
                                message="answer is not grounded",
                                metrics=[Metric(name="confidence", value=0.2)],
                                details={"check_name": "Groundedness"},
                            )
                        ],
                        duration_ms=120,
                    )
                ],
                duration_ms=120,
                final_trace=Trace(),
            ),
        ],
        duration_ms=220,
    )


def test_export_emits_suite_scenario_check_and_trace_events() -> None:
    tracer = FakeTracer()
    meter = FakeMeter()
    exporter = OTelExporter(tracer=tracer, meter=meter)

    exporter.export(_sample_suite_result())

    assert [span.name for span in tracer.spans] == [
        "giskard.suite",
        "giskard.scenario",
        "giskard.scenario",
    ]

    suite_span = tracer.spans[0]
    assert suite_span.attributes["giskard.suite.status"] == "fail"
    assert suite_span.attributes["giskard.suite.duration_ms"] == 220
    assert suite_span.attributes["giskard.suite.pass_rate"] == 0.5
    assert suite_span.events == []

    scenario_span = tracer.spans[1]
    interaction_events = [
        event for event in scenario_span.events if event[0] == "giskard.trace.interaction"
    ]
    check_events = [event for event in scenario_span.events if event[0] == "giskard.check"]
    assert scenario_span.attributes["giskard.trace.interaction_count"] == 1
    assert interaction_events[0][1]["giskard.trace.interaction.index"] == 1
    assert "What is Giskard?" in interaction_events[0][1][
        "giskard.trace.interaction.inputs"
    ]
    assert check_events[0][1]["giskard.check.name"] == "Groundedness"


def test_export_can_hide_trace_payloads() -> None:
    tracer = FakeTracer()
    meter = FakeMeter()
    exporter = OTelExporter(
        tracer=tracer,
        meter=meter,
        include_trace_payloads=False,
    )

    exporter.export(_sample_suite_result())

    interaction_event = tracer.spans[1].events[0]
    assert interaction_event[0] == "giskard.trace.interaction"
    assert "giskard.trace.interaction.inputs" not in interaction_event[1]
    assert "giskard.trace.interaction.outputs" not in interaction_event[1]
    assert (
        interaction_event[1]["giskard.trace.interaction.metadata.model"]
        == "test-model"
    )


def test_export_records_suite_scenario_check_and_score_metrics() -> None:
    tracer = FakeTracer()
    meter = FakeMeter()
    exporter = OTelExporter(tracer=tracer, meter=meter)

    exporter.export(_sample_suite_result())

    assert meter.instruments["giskard.suite.duration"].records[0][1] == 220
    assert meter.instruments["giskard.suite.pass_rate"].records[0][1] == 0.5
    assert [
        record[1]
        for record in meter.instruments["giskard.scenario.duration"].records
    ] == [
        100,
        120,
    ]
    assert len(meter.instruments["giskard.check.count"].records) == 3

    score_records = meter.instruments["giskard.check.metric.value"].records
    assert [record[1] for record in score_records] == [0.95, 0.2]
    assert score_records[0][2]["giskard.check.metric.name"] == "score"
    assert score_records[1][2]["giskard.check.metric.name"] == "confidence"


def test_export_with_opentelemetry_sdk_in_memory_exporters() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    exporter = OTelExporter(
        tracer=tracer_provider.get_tracer(__name__),
        meter=meter_provider.get_meter(__name__),
    )

    exporter.export(_sample_suite_result())

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "giskard.scenario",
        "giskard.scenario",
        "giskard.suite",
    ]
    assert spans[-1].attributes["giskard.suite.pass_rate"] == 0.5
    assert any(
        event.name == "giskard.trace.interaction" for event in spans[0].events
    )
    assert any(event.name == "giskard.check" for event in spans[0].events)
    assert not any(event.name == "giskard.check" for event in spans[-1].events)

    metrics_data = metric_reader.get_metrics_data()
    metric_names = {
        metric.name
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }
    assert {
        "giskard.suite.duration",
        "giskard.suite.pass_rate",
        "giskard.scenario.duration",
        "giskard.check.count",
        "giskard.check.metric.value",
    }.issubset(metric_names)

    tracer_provider.shutdown()
    meter_provider.shutdown()


def test_flush_and_shutdown_support_providers_without_timeout_argument() -> None:
    exporter = OTelExporter(tracer=FakeTracer(), meter=FakeMeter())
    provider = FakeProvider(supports_timeout=False)
    meter_provider = FakeProvider()
    exporter._provider = provider
    exporter._meter_provider = meter_provider

    assert exporter.force_flush(timeout_millis=10_000) is True
    exporter.shutdown(timeout_millis=10_000)

    assert provider.flushed is True
    assert provider.shutdown_called is True
    assert provider.timeout_millis is None
    assert meter_provider.flushed is True
    assert meter_provider.shutdown_called is True
    assert meter_provider.timeout_millis == 10_000


def test_exporter_without_injected_otel_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"giskard-checks\[otel\]"):
        OTelExporter(endpoint="http://localhost:4317")
