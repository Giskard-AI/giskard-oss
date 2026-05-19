from asyncio import TaskGroup
from typing import Any

import numpy as np

from ..core.interaction import Trace
from ..core.scenario import Scenario
from ..scenarios.suite import Suite
from .base import ScenarioGenerator
from .registry import _normalize_generator, suite_generator_registry


def _normalize_generators(
    generators: list[ScenarioGenerator | type[ScenarioGenerator]],
) -> list[ScenarioGenerator]:
    return [_normalize_generator(g) for g in generators]


async def _generate_scenarios(
    description: str,
    languages: list[str],
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    rng = np.random.default_rng(seed)

    tasks = []
    async with TaskGroup() as task_group:
        if max_scenarios is not None and len(generators) > 0:
            counts = rng.multinomial(
                max_scenarios, np.ones(len(generators)) / len(generators)
            )
            for generator, n in zip(generators, counts):
                tasks.append(
                    task_group.create_task(
                        generator.generate_scenario(description, languages, int(n), rng)
                    )
                )
        else:
            for generator in generators:
                tasks.append(
                    task_group.create_task(
                        generator.generate_scenario(description, languages)
                    )
                )

    return [scenario for task in tasks for scenario in task.result()]


async def generate_suite(
    description: str,
    languages: list[str],
    generators: list[ScenarioGenerator | type[ScenarioGenerator]] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
) -> Suite[Any, Any]:
    resolved = (
        _normalize_generators(generators)
        if generators is not None
        else suite_generator_registry.generators()
    )
    scenarios = await _generate_scenarios(
        description, languages, resolved, max_scenarios, seed
    )
    return Suite(name="Scenarios", scenarios=scenarios)
