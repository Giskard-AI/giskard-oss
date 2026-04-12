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
from giskard.checks.export.langfuse import LangfuseExporter


class FakeObservation:
    def __init__(self, trace_id: str, observation_id: str) -> None:
        self.trace_id = trace_id
        self.id = observation_id

    def __enter__(self) -> "FakeObservation":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.flushed = False

    def create_trace_id(self) -> str:
        return "trace-1"

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservation:
        observation_id = f"observation-{len(self.observations) + 1}"
        self.observations.append({"id": observation_id, **kwargs})
        return FakeObservation(trace_id="trace-1", observation_id=observation_id)

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


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
                            )
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
            )
        ],
        duration_ms=100,
    )


def test_langfuse_export_maps_suite_scenarios_generations_and_scores() -> None:
    client = FakeLangfuseClient()
    exporter = LangfuseExporter(client=client)

    exporter.export(_sample_suite_result())

    assert [observation["name"] for observation in client.observations] == [
        "giskard.suite",
        "scenario_pass",
        "interaction_1",
        "Groundedness",
    ]
    assert client.observations[0]["as_type"] == "evaluator"
    assert client.observations[2]["as_type"] == "generation"
    assert client.observations[2]["input"] == {"question": "What is Giskard?"}
    assert client.observations[2]["output"] == {
        "answer": "An evaluation framework."
    }
    assert client.observations[2]["model"] == "test-model"

    assert [score["name"] for score in client.scores] == [
        "giskard.suite.pass_rate",
        "Groundedness",
        "Groundedness.score",
    ]
    assert client.scores[0]["value"] == 1.0
    assert client.scores[1]["observation_id"] == "observation-4"
    assert client.scores[1]["value"] == 0.95
    assert client.scores[2]["value"] == 0.95


def test_langfuse_export_can_hide_trace_payloads() -> None:
    client = FakeLangfuseClient()
    exporter = LangfuseExporter(client=client, include_trace_payloads=False)

    exporter.export(_sample_suite_result())

    generation = client.observations[2]
    assert generation["as_type"] == "generation"
    assert generation["input"] is None
    assert generation["output"] is None
    assert generation["metadata"]["model"] == "test-model"


def test_langfuse_flush_delegates_to_client() -> None:
    client = FakeLangfuseClient()
    exporter = LangfuseExporter(client=client)

    exporter.flush()

    assert client.flushed is True


def test_langfuse_exporter_without_dependency_raises_helpful_error(
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
        if name == "langfuse" or name.startswith("langfuse."):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"giskard-checks\[langfuse\]"):
        LangfuseExporter(public_key="pk", secret_key="sk")
