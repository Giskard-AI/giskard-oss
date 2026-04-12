from typing import Any

import pytest
from giskard.checks import CheckResult, Scenario, Suite, from_fn
from giskard.checks.export.monitoring import evaluate_production_sample


class FakeExporter:
    def __init__(self) -> None:
        self.results: list[Any] = []

    def export(self, result: Any) -> None:
        self.results.append(result)


@pytest.mark.asyncio
async def test_evaluate_production_sample_exports_and_alerts() -> None:
    exporter = FakeExporter()
    suite = Suite(name="production")
    suite.append(
        Scenario("failing").check(
            from_fn(
                lambda trace: CheckResult.failure(message="bad answer"),
                name="failing_check",
            )
        )
    )

    result = await evaluate_production_sample(
        suite,
        exporters=[exporter],
        sample_rate=1.0,
        min_pass_rate=0.9,
    )

    assert result.sampled is True
    assert result.result is not None
    assert result.result.pass_rate == 0.0
    assert result.alert is not None
    assert result.alert.threshold == 0.9
    assert exporter.results == [result.result]


@pytest.mark.asyncio
async def test_evaluate_production_sample_can_skip_unsampled_request() -> None:
    suite = Suite(name="production")
    suite.append(Scenario("passing").check(from_fn(lambda trace: True, name="ok")))

    result = await evaluate_production_sample(suite, sample_rate=0.0)

    assert result.sampled is False
    assert result.result is None
    assert result.alert is None


@pytest.mark.asyncio
async def test_evaluate_production_sample_validates_thresholds() -> None:
    suite = Suite(name="production")

    with pytest.raises(ValueError, match="sample_rate"):
        _ = await evaluate_production_sample(suite, sample_rate=1.5)

    with pytest.raises(ValueError, match="min_pass_rate"):
        _ = await evaluate_production_sample(suite, min_pass_rate=-0.1)
