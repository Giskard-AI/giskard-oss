"""Offline scan stub — exercises generate_suite with no network generators."""

from typing import Any

import pytest
from giskard.checks import Equals, Scenario, SuiteResult
from giskard.scan.catalog import generate_suite


async def echo(inputs: str) -> str:
    return inputs


async def test_generate_suite_empty_generators_offline() -> None:
    suite = await generate_suite(
        description="Demo support agent",
        languages=["en"],
        generators=[],
        max_scenarios=5,
    )
    assert suite.scenarios == []


async def test_run_static_scenario_as_scan_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = (
        Scenario("stub")
        .interact(inputs="ping", outputs=echo)
        .check(Equals(key="trace.last.outputs", expected_value="ping"))
    )

    class _FakeSuite:
        scenarios = [scenario]

        async def run(
            self,
            target: object,
            parallel: bool = True,
            max_concurrency: int | None = None,
            return_exception: bool = False,
        ) -> SuiteResult:
            _ = parallel, max_concurrency, return_exception
            scenario_result = await scenario.run(target=target)  # pyright: ignore[reportArgumentType]
            return SuiteResult(
                results=[scenario_result],
                duration_ms=scenario_result.duration_ms,
            )

    async def fake_generate_suite(**kwargs: Any) -> _FakeSuite:
        _ = kwargs
        return _FakeSuite()

    import giskard.scan.vulnerability as vulnerability_module

    monkeypatch.setattr(vulnerability_module, "generate_suite", fake_generate_suite)

    from giskard.scan import vulnerability_scan

    result = await vulnerability_scan(
        target=echo,
        description="Demo agent",
        languages=["en"],
        max_scenarios=1,
    )
    assert result.pass_rate == 1.0
