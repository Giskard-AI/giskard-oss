import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..core.result import CheckResult, ScenarioResult, SuiteResult

_OTEL_INSTALL_HINT = (
    "OpenTelemetry export requires optional dependencies. "
    "Install them with `pip install giskard-checks[otel]`."
)


def _status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value)


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


def _attribute_value(value: Any) -> str | bool | int | float:
    if isinstance(value, str | bool | int | float):
        return value
    if value is None:
        return ""
    return _json(value)


def _details_attributes(details: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(key, str):
            attributes[f"{prefix}.{key}"] = _attribute_value(value)
    return attributes


def _check_label(result: CheckResult, fallback: str) -> str:
    return str(
        result.details.get("check_name")
        or result.details.get("check_kind")
        or result.details.get("name")
        or fallback
    )


def _iter_checks(
    scenario: ScenarioResult[Any],
) -> Iterable[tuple[int, int, CheckResult]]:
    for step_index, step in enumerate(scenario.steps, start=1):
        for check_index, check in enumerate(step.results, start=1):
            yield step_index, check_index, check


def _suite_status(result: SuiteResult) -> str:
    if result.errored_count:
        return "error"
    if result.failed_count:
        return "fail"
    if result.results and result.skipped_count == len(result.results):
        return "skip"
    return "pass"


def _suite_attributes(result: SuiteResult) -> dict[str, Any]:
    return {
        "giskard.suite.status": _suite_status(result),
        "giskard.suite.duration_ms": result.duration_ms,
        "giskard.suite.scenario_count": len(result.results),
        "giskard.suite.passed_count": result.passed_count,
        "giskard.suite.failed_count": result.failed_count,
        "giskard.suite.errored_count": result.errored_count,
        "giskard.suite.skipped_count": result.skipped_count,
        "giskard.suite.pass_rate": result.pass_rate,
    }


def _scenario_attributes(
    scenario: ScenarioResult[Any],
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "giskard.scenario.index": index,
        "giskard.scenario.name": scenario.scenario_name,
        "giskard.scenario.status": _status_value(scenario.status),
        "giskard.scenario.duration_ms": scenario.duration_ms,
        "giskard.scenario.step_count": len(scenario.steps),
        "giskard.scenario.check_count": sum(
            len(step.results) for step in scenario.steps
        ),
        "giskard.trace.interaction_count": len(scenario.final_trace.interactions),
    }


def _check_attributes(
    result: CheckResult,
    *,
    scenario: ScenarioResult[Any],
    scenario_index: int,
    step_index: int,
    check_index: int,
) -> dict[str, Any]:
    attributes = {
        "giskard.scenario.index": scenario_index,
        "giskard.scenario.name": scenario.scenario_name,
        "giskard.step.index": step_index,
        "giskard.check.index": check_index,
        "giskard.check.name": _check_label(result, f"check_{check_index}"),
        "giskard.check.status": _status_value(result.status),
        "giskard.check.message": result.message or "",
    }
    attributes.update(_details_attributes(result.details, "giskard.check.details"))
    return attributes


def _interaction_attributes(
    interaction: Any,
    *,
    scenario: ScenarioResult[Any],
    scenario_index: int,
    interaction_index: int,
    include_payloads: bool,
) -> dict[str, Any]:
    attributes = {
        "giskard.scenario.index": scenario_index,
        "giskard.scenario.name": scenario.scenario_name,
        "giskard.trace.interaction.index": interaction_index,
    }

    metadata = getattr(interaction, "metadata", {})
    if isinstance(metadata, Mapping):
        attributes.update(
            _details_attributes(metadata, "giskard.trace.interaction.metadata")
        )

    if include_payloads:
        attributes.update(
            {
                "giskard.trace.interaction.inputs": _attribute_value(
                    getattr(interaction, "inputs", None)
                ),
                "giskard.trace.interaction.outputs": _attribute_value(
                    getattr(interaction, "outputs", None)
                ),
            }
        )

    return attributes


def _require_otel_sdk() -> dict[str, Any]:
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ImportError(_OTEL_INSTALL_HINT) from exc

    return {
        "metrics": metrics,
        "trace": trace,
        "OTLPMetricExporter": OTLPMetricExporter,
        "OTLPSpanExporter": OTLPSpanExporter,
        "MeterProvider": MeterProvider,
        "PeriodicExportingMetricReader": PeriodicExportingMetricReader,
        "Resource": Resource,
        "TracerProvider": TracerProvider,
        "BatchSpanProcessor": BatchSpanProcessor,
    }


def _call_provider_method(method: Any, *, timeout_millis: int) -> Any:
    try:
        return method(timeout_millis=timeout_millis)
    except TypeError as exc:
        if "timeout_millis" not in str(exc):
            raise
        return method()


class OTelExporter:
    """Export suite results as OpenTelemetry traces and metrics."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        service_name: str = "giskard-checks",
        tracer: Any | None = None,
        meter: Any | None = None,
        set_global_provider: bool = False,
        include_trace_payloads: bool = True,
    ) -> None:
        self._provider = None
        self._meter_provider = None
        self._include_trace_payloads = include_trace_payloads

        if tracer is None or meter is None:
            sdk = _require_otel_sdk()
            resource = sdk["Resource"].create({"service.name": service_name})

            if tracer is None:
                span_exporter = sdk["OTLPSpanExporter"](endpoint=endpoint)
                self._provider = sdk["TracerProvider"](resource=resource)
                self._provider.add_span_processor(
                    sdk["BatchSpanProcessor"](span_exporter)
                )
                if set_global_provider:
                    sdk["trace"].set_tracer_provider(self._provider)
                    tracer = sdk["trace"].get_tracer(__name__)
                else:
                    tracer = self._provider.get_tracer(__name__)

            if meter is None:
                metric_exporter = sdk["OTLPMetricExporter"](endpoint=endpoint)
                metric_reader = sdk["PeriodicExportingMetricReader"](metric_exporter)
                self._meter_provider = sdk["MeterProvider"](
                    resource=resource,
                    metric_readers=[metric_reader],
                )
                if set_global_provider:
                    sdk["metrics"].set_meter_provider(self._meter_provider)
                    meter = sdk["metrics"].get_meter(__name__)
                else:
                    meter = self._meter_provider.get_meter(__name__)

        self._tracer = tracer
        self._meter = meter
        self._suite_duration = meter.create_histogram(
            "giskard.suite.duration",
            unit="ms",
            description="Suite execution duration.",
        )
        self._suite_pass_rate = meter.create_histogram(
            "giskard.suite.pass_rate",
            unit="1",
            description="Suite pass rate excluding skipped scenarios.",
        )
        self._scenario_duration = meter.create_histogram(
            "giskard.scenario.duration",
            unit="ms",
            description="Scenario execution duration.",
        )
        self._check_count = meter.create_counter(
            "giskard.check.count",
            unit="1",
            description="Number of executed checks by status.",
        )
        self._check_metric_value = meter.create_histogram(
            "giskard.check.metric.value",
            unit="1",
            description="Numeric metric values emitted by checks.",
        )

    def export(self, result: SuiteResult) -> None:
        """Export a suite result to the configured OpenTelemetry backend."""
        suite_attributes = _suite_attributes(result)

        with self._tracer.start_as_current_span(
            "giskard.suite",
            attributes=suite_attributes,
        ):
            self._record_suite_metrics(result, suite_attributes)

            for scenario_index, scenario in enumerate(result.results, start=1):
                self._export_scenario(
                    scenario=scenario,
                    scenario_index=scenario_index,
                )

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush configured OpenTelemetry providers when they expose force_flush."""
        ok = True
        for provider in (self._provider, self._meter_provider):
            force_flush = getattr(provider, "force_flush", None)
            if force_flush is not None:
                ok = bool(
                    _call_provider_method(
                        force_flush,
                        timeout_millis=timeout_millis,
                    )
                ) and ok
        return ok

    def shutdown(self, timeout_millis: int = 30_000) -> None:
        """Shutdown configured OpenTelemetry providers when they expose shutdown."""
        for provider in (self._meter_provider, self._provider):
            shutdown = getattr(provider, "shutdown", None)
            if shutdown is not None:
                _call_provider_method(shutdown, timeout_millis=timeout_millis)

    def _export_scenario(
        self,
        *,
        scenario: ScenarioResult[Any],
        scenario_index: int,
    ) -> None:
        scenario_attributes = _scenario_attributes(
            scenario,
            index=scenario_index,
        )
        self._scenario_duration.record(
            scenario.duration_ms,
            attributes=scenario_attributes,
        )

        with self._tracer.start_as_current_span(
            "giskard.scenario",
            attributes=scenario_attributes,
        ) as scenario_span:
            for interaction_index, interaction in enumerate(
                scenario.final_trace.interactions,
                start=1,
            ):
                scenario_span.add_event(
                    "giskard.trace.interaction",
                    attributes=_interaction_attributes(
                        interaction,
                        scenario=scenario,
                        scenario_index=scenario_index,
                        interaction_index=interaction_index,
                        include_payloads=self._include_trace_payloads,
                    ),
                )

            for step_index, check_index, check in _iter_checks(scenario):
                check_attributes = _check_attributes(
                    check,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    step_index=step_index,
                    check_index=check_index,
                )
                scenario_span.add_event("giskard.check", attributes=check_attributes)
                self._record_check_metrics(check, check_attributes)

    def _record_suite_metrics(
        self,
        result: SuiteResult,
        attributes: dict[str, Any],
    ) -> None:
        self._suite_duration.record(result.duration_ms, attributes=attributes)
        self._suite_pass_rate.record(result.pass_rate, attributes=attributes)

    def _record_check_metrics(
        self,
        result: CheckResult,
        attributes: dict[str, Any],
    ) -> None:
        self._check_count.add(1, attributes=attributes)
        for metric in result.metrics:
            self._check_metric_value.record(
                metric.value,
                attributes={
                    **attributes,
                    "giskard.check.metric.name": metric.name,
                },
            )
