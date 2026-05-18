from asyncio import TaskGroup
from typing import Any

import numpy as np

from ..core.interaction import Trace
from ..core.scenario import Scenario
from ..scenarios.suite import Suite
from .adversarial_generator import AdversarialScenarioGenerator
from .base import ScenarioGenerator

GENERATORS: list[ScenarioGenerator] = [
    AdversarialScenarioGenerator(),
]


async def _generate_scenarios(
    description: str,
    languages: list[str],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    tasks = []
    async with TaskGroup() as task_group:
        for generator in GENERATORS:
            tasks.append(
                task_group.create_task(
                    generator.generate_scenario(description, languages)
                )
            )

    scenarios = [scenario for task in tasks for scenario in task.result()]

    if max_scenarios is None:
        return scenarios

    rng = np.random.default_rng(seed)

    if max_scenarios >= len(scenarios):
        return scenarios

    return rng.choice(scenarios, size=max_scenarios, replace=False).tolist()


async def generate_suite(
    description: str,
    languages: list[str],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> Suite[Any, Any]:
    scenarios = await _generate_scenarios(description, languages, max_scenarios, seed)
    return Suite(name="Scenarios", scenarios=scenarios)
