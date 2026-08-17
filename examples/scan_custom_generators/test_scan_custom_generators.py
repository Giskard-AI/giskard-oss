"""Offline scan example with a custom scenario generator."""

from typing import Any

import numpy as np
from giskard.checks import Equals, Scenario, Trace
from giskard.scan import ScenarioGenerator, TargetMode, generate_suite
from giskard.scan.generators.base import DEFAULT_TARGET_MODE, ScenarioContext


async def echo(inputs: str) -> str:
    return inputs


class StaticEchoScenarioGenerator(ScenarioGenerator):
    """Generate a deterministic scenario without a model provider."""

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = DEFAULT_TARGET_MODE,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Build one scenario and honor the suite-wide scenario budget."""
        _ = context, rng, target_mode
        if max_scenarios == 0:
            return []
        return [
            Scenario("echo")
            .interact(inputs="ping", outputs=echo)
            .check(Equals(target_key="trace.last.outputs", expected_value="ping"))
        ]


async def test_run_custom_generator_offline() -> None:
    suite = await generate_suite(
        description="Demo support agent",
        languages=["en"],
        generators=[StaticEchoScenarioGenerator()],
        max_scenarios=1,
    )

    result = await suite.run(echo)

    assert len(suite.scenarios) == 1
    assert result.pass_rate == 1.0
