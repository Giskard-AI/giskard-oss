from typing import Any

import numpy as np
import pytest
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.scan.catalog import generate_suite
from giskard.scan.generators.base import ScenarioGenerator
from giskard.scan.vulnerability import vulnerability_suite_generator_registry


class _StubGenerator(ScenarioGenerator):
    """Returns a fixed number of scenarios for testing."""

    name: str = "stub"
    scenario_count: int = 1

    async def generate_scenario(
        self,
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        n = max_scenarios if max_scenarios is not None else self.scenario_count
        return [Scenario(name=f"stub-{self.name}-{i}") for i in range(n)]


async def test_generate_suite_requires_explicit_generators():
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'generators'"
    ):
        await generate_suite(  # pyright: ignore[reportCallIssue]
            "My chatbot", languages=["en"]
        )


async def test_generate_suite_does_not_read_vulnerability_registry(
    isolated_vulnerability_registry,
):
    vulnerability_suite_generator_registry.register(_StubGenerator(name="registry"))
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator(name="explicit")],
    )
    assert [scenario.name for scenario in suite.scenarios] == ["stub-explicit-0"]


async def test_generate_suite_generators_bare_type_is_normalized():
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator],
    )
    assert len(suite.scenarios) == 1
    assert suite.scenarios[0].name == "stub-stub-0"


async def test_generate_suite_empty_generators_returns_empty_suite(
    isolated_vulnerability_registry,
):
    vulnerability_suite_generator_registry.register(_StubGenerator(name="a"))
    suite = await generate_suite("My chatbot", languages=["en"], generators=[])
    assert suite.scenarios == []


async def test_generate_suite_max_scenarios_limits_output():
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[
            _StubGenerator(name="a", scenario_count=5),
            _StubGenerator(name="b", scenario_count=5),
        ],
        max_scenarios=2,
    )
    assert len(suite.scenarios) == 2


async def test_generate_suite_max_scenarios_distributed_across_generators():
    """Budget is split across generators; each receives a non-negative count summing to max_scenarios."""
    received: dict[str, int | None] = {}

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            description: str,
            languages: list[str],
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            received[self.name] = max_scenarios
            n = max_scenarios if max_scenarios is not None else 0
            return [Scenario(name=f"{self.name}-{i}") for i in range(n)]

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_TrackingGenerator(name="x"), _TrackingGenerator(name="y")],
        max_scenarios=4,
        seed=42,
    )
    assert received["x"] is not None
    assert received["y"] is not None
    assert received["x"] + received["y"] == 4


async def test_generate_suite_no_max_passes_none_to_generators():
    """Without max_scenarios, generators receive max_scenarios=None."""
    received: dict[str, int | None] = {}

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            description: str,
            languages: list[str],
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            received[self.name] = max_scenarios
            return []

    await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_TrackingGenerator(name="z")],
    )
    assert received["z"] is None


async def test_generate_suite_registry_generators_not_mutated():
    """The catalog must not mutate provided generator instances."""
    gen = _StubGenerator(name="orig", scenario_count=3)
    original_count = gen.scenario_count

    await generate_suite(
        "My chatbot", languages=["en"], generators=[gen], max_scenarios=1
    )

    assert gen.scenario_count == original_count


async def test_generate_suite_negative_max_scenarios_raises_valueerror():
    """max_scenarios < 0 raises ValueError."""
    with pytest.raises(ValueError, match="max_scenarios must be non-negative, got -1"):
        await generate_suite(
            "My chatbot", languages=["en"], generators=[], max_scenarios=-1
        )


async def test_generate_suite_max_scenarios_zero_returns_empty():
    """max_scenarios=0 is a valid no-op budget: returns an empty suite."""
    suite = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=[_StubGenerator(name="a", scenario_count=5)],
        max_scenarios=0,
    )
    assert suite.scenarios == []


async def test_generate_suite_reproducibility():
    """Same seed produces identical per-generator scenario name allocation."""

    class _TrackingGenerator(ScenarioGenerator):
        name: str

        async def generate_scenario(
            self,
            description: str,
            languages: list[str],
            max_scenarios: int | None = None,
            rng: np.random.Generator | None = None,
        ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
            return [
                Scenario(name=f"{self.name}-{i}") for i in range(max_scenarios or 0)
            ]

    generators = [_TrackingGenerator(name="p"), _TrackingGenerator(name="q")]

    suite_a = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=generators,
        max_scenarios=10,
        seed=99,
    )
    suite_b = await generate_suite(
        "My chatbot",
        languages=["en"],
        generators=generators,
        max_scenarios=10,
        seed=99,
    )

    assert [s.name for s in suite_a.scenarios] == [s.name for s in suite_b.scenarios]
