import importlib
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
from giskard.checks import Equals, Scenario, Trace
from giskard.checks.scenarios.suite import Suite
from giskard.scan.generators.base import ScenarioContext, ScenarioGenerator
from giskard.scan.quality import quality_scan, quality_suite_generator_registry
from giskard.scan.vulnerability import (
    vulnerability_scan,
    vulnerability_suite_generator_registry,
)

_TELEMETRY_MODULES = (
    "giskard.scan.quality",
    "giskard.scan.vulnerability",
    "giskard.scan.integrations._entry_point",
)


class _DeterministicGenerator(ScenarioGenerator):
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: str = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        _ = context, rng, target_mode
        scenarios = [
            Scenario("passes")
            .interact("ok")
            .check(Equals(expected_value="ok", key="trace.last.outputs")),
            Scenario("fails")
            .interact("bad")
            .check(Equals(expected_value="good", key="trace.last.outputs")),
        ]
        return scenarios if max_scenarios is None else scenarios[:max_scenarios]


@pytest.fixture
def telemetry_capture(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, dict[str, Any]]] = []

    def capture(event: str, *, properties: dict[str, Any]) -> None:
        events.append((event, properties))

    for module_name in _TELEMETRY_MODULES:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "telemetry_capture", capture)
        monkeypatch.setattr(module, "telemetry_run_context", nullcontext)
        monkeypatch.setattr(module, "telemetry_tag", lambda *args, **kwargs: None)

    return events


@pytest.mark.usefixtures("isolated_quality_registry")
async def test_quality_scan_telemetry(telemetry_capture):
    quality_suite_generator_registry.register(_DeterministicGenerator())

    async def target(inputs: str) -> str:
        return inputs

    await quality_scan(
        target=target,
        description="Support agent",
        languages=["en", "fr"],
        knowledge_base=["reference document"],
        max_scenarios=2,
        parallel=False,
    )

    assert telemetry_capture[0][0] == "scan_quality_run_started"
    started = telemetry_capture[0][1]
    assert started["integration"] == "giskard-scan"
    assert started["scan_kind"] == "quality"
    assert started["language_count"] == 2
    assert started["generator_count"] == 1
    assert started["scenario_count"] == 2
    assert started["has_knowledge_base"] is True
    assert started["parallel"] is False

    assert telemetry_capture[1][0] == "scan_quality_run_finished"
    finished = telemetry_capture[1][1]
    assert finished["passed_count"] == 1
    assert finished["failed_count"] == 1
    assert "duration_ms" in finished


@pytest.mark.usefixtures("isolated_vulnerability_registry")
async def test_vulnerability_scan_telemetry(telemetry_capture):
    vulnerability_suite_generator_registry.register(_DeterministicGenerator())

    async def target(inputs: str) -> str:
        return inputs

    await vulnerability_scan(
        target=target,
        description="Chatbot",
        languages=["en"],
        parallel=True,
        commercial_use=True,
    )

    assert telemetry_capture[0][0] == "scan_vulnerability_run_started"
    started = telemetry_capture[0][1]
    assert started["scan_kind"] == "vulnerability"
    assert started["commercial_use"] is True
    assert started["parallel"] is True

    assert telemetry_capture[1][0] == "scan_vulnerability_run_finished"
    finished = telemetry_capture[1][1]
    assert finished["passed_count"] == 1
    assert finished["failed_count"] == 1


async def test_third_party_scan_telemetry_garak(telemetry_capture, monkeypatch):
    class _FakeGarakAdapter:
        async def run(self, target, **kwargs):
            _ = kwargs
            suite = Suite(name="garak")
            suite.append(
                Scenario("probe")
                .interact("prompt")
                .check(Equals(expected_value="prompt", key="trace.last.outputs"))
            )
            return await suite.run(target)

    monkeypatch.setattr(
        "giskard.scan.integrations.garak.GarakScanAdapter",
        _FakeGarakAdapter,
    )

    async def target(inputs: str) -> str:
        return inputs

    entry_point = importlib.import_module("giskard.scan.integrations._entry_point")
    await entry_point.third_party_scan(
        target,
        tool="garak",
        description="A test agent",
        probes=["probe.one"],
        target_mode="singleturn",
    )

    assert telemetry_capture[0][0] == "scan_third_party_run_started"
    started = telemetry_capture[0][1]
    assert started["scan_kind"] == "third_party_garak"
    assert started["tool"] == "garak"
    assert started["has_probe_filter"] is True
    assert started["target_mode"] == "singleturn"

    assert telemetry_capture[1][0] == "scan_third_party_run_finished"
    finished = telemetry_capture[1][1]
    assert finished["scenario_count"] == 1
    assert finished["passed_count"] == 1
