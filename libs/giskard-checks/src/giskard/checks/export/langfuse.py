import json
from typing import Any

from ..core.result import CheckResult, ScenarioResult, SuiteResult

_LANGFUSE_INSTALL_HINT = (
    "Langfuse export requires optional dependencies. "
    "Install them with `pip install giskard-checks[langfuse]`."
)


def _require_langfuse() -> type[Any]:
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise ImportError(_LANGFUSE_INSTALL_HINT) from exc
    return Langfuse


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


def _check_label(result: CheckResult, fallback: str) -> str:
    return str(
        result.details.get("check_name")
        or result.details.get("check_kind")
        or result.details.get("name")
        or fallback
    )


def _score_value(result: CheckResult) -> float | str:
    if result.metrics:
        return result.metrics[0].value
    if result.passed:
        return 1.0
    if result.skipped:
        return "skipped"
    return 0.0


def _score_data_type(result: CheckResult) -> str:
    if result.metrics:
        return "NUMERIC"
    if result.skipped:
        return "CATEGORICAL"
    return "NUMERIC"


def _score_comment(result: CheckResult) -> str | None:
    if result.message:
        return result.message
    if result.metrics:
        metric = result.metrics[0]
        return f"{metric.name}={metric.value}"
    return None


def _metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": _attribute_value(value)}


class LangfuseExporter:
    """Export suite results to Langfuse traces, observations, and scores."""

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        client: Any | None = None,
        sample_rate: float | None = None,
        include_trace_payloads: bool = True,
    ) -> None:
        if client is None:
            Langfuse = _require_langfuse()
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
            )

        self._client = client
        self._include_trace_payloads = include_trace_payloads

    def export(self, result: SuiteResult) -> None:
        """Export a suite result to Langfuse."""
        trace_id = self._client.create_trace_id()
        suite_metadata = {
            "giskard.suite.duration_ms": result.duration_ms,
            "giskard.suite.scenario_count": len(result.results),
            "giskard.suite.passed_count": result.passed_count,
            "giskard.suite.failed_count": result.failed_count,
            "giskard.suite.errored_count": result.errored_count,
            "giskard.suite.skipped_count": result.skipped_count,
            "giskard.suite.pass_rate": result.pass_rate,
        }

        with self._client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="giskard.suite",
            as_type="evaluator",
            metadata=suite_metadata,
            level="ERROR" if result.errored_count else "DEFAULT",
            status_message=f"pass_rate={result.pass_rate}",
        ) as suite_observation:
            self._client.create_score(
                trace_id=suite_observation.trace_id,
                observation_id=suite_observation.id,
                name="giskard.suite.pass_rate",
                value=result.pass_rate,
                data_type="NUMERIC",
                metadata=suite_metadata,
            )

            for scenario_index, scenario in enumerate(result.results, start=1):
                self._export_scenario(
                    scenario=scenario,
                    scenario_index=scenario_index,
                )

    def flush(self) -> None:
        """Flush pending Langfuse telemetry."""
        flush = getattr(self._client, "flush", None)
        if flush is not None:
            flush()

    def _export_scenario(
        self,
        *,
        scenario: ScenarioResult[Any],
        scenario_index: int,
    ) -> None:
        scenario_metadata = {
            "giskard.scenario.index": scenario_index,
            "giskard.scenario.status": _status_value(scenario.status),
            "giskard.scenario.duration_ms": scenario.duration_ms,
            "giskard.scenario.step_count": len(scenario.steps),
            "giskard.scenario.check_count": sum(
                len(step.results) for step in scenario.steps
            ),
            "giskard.trace.interaction_count": len(scenario.final_trace.interactions),
        }

        with self._client.start_as_current_observation(
            name=scenario.scenario_name,
            as_type="evaluator",
            metadata=scenario_metadata,
            level="ERROR" if scenario.errored else "DEFAULT",
            status_message=_status_value(scenario.status),
        ) as scenario_observation:
            for interaction_index, interaction in enumerate(
                scenario.final_trace.interactions,
                start=1,
            ):
                self._export_interaction(
                    interaction=interaction,
                    interaction_index=interaction_index,
                )

            for step_index, step in enumerate(scenario.steps, start=1):
                for check_index, check in enumerate(step.results, start=1):
                    self._export_check(
                        check=check,
                        scenario=scenario,
                        scenario_observation=scenario_observation,
                        step_index=step_index,
                        check_index=check_index,
                    )

    def _export_interaction(
        self,
        *,
        interaction: Any,
        interaction_index: int,
    ) -> None:
        metadata = {
            "giskard.trace.interaction.index": interaction_index,
            **_metadata(getattr(interaction, "metadata", {})),
        }
        interaction_metadata = getattr(interaction, "metadata", {})
        model = (
            interaction_metadata.get("model")
            if isinstance(interaction_metadata, dict)
            else None
        )

        with self._client.start_as_current_observation(
            name=f"interaction_{interaction_index}",
            as_type="generation",
            input=getattr(interaction, "inputs", None)
            if self._include_trace_payloads
            else None,
            output=getattr(interaction, "outputs", None)
            if self._include_trace_payloads
            else None,
            metadata=metadata,
            model=model,
        ):
            pass

    def _export_check(
        self,
        *,
        check: CheckResult,
        scenario: ScenarioResult[Any],
        scenario_observation: Any,
        step_index: int,
        check_index: int,
    ) -> None:
        name = _check_label(check, f"check_{check_index}")
        metadata = {
            "giskard.scenario.name": scenario.scenario_name,
            "giskard.step.index": step_index,
            "giskard.check.index": check_index,
            "giskard.check.status": _status_value(check.status),
            "giskard.check.details": _json(check.details),
        }

        with self._client.start_as_current_observation(
            name=name,
            as_type="evaluator",
            metadata=metadata,
            level="ERROR" if check.errored else "DEFAULT",
            status_message=check.message or _status_value(check.status),
        ) as check_observation:
            self._client.create_score(
                trace_id=scenario_observation.trace_id,
                observation_id=check_observation.id,
                name=name,
                value=_score_value(check),
                data_type=_score_data_type(check),
                comment=_score_comment(check),
                metadata=metadata,
            )

            for metric in check.metrics:
                self._client.create_score(
                    trace_id=scenario_observation.trace_id,
                    observation_id=check_observation.id,
                    name=f"{name}.{metric.name}",
                    value=metric.value,
                    data_type="NUMERIC",
                    metadata=metadata,
                )
