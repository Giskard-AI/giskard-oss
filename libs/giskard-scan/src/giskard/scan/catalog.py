from asyncio import TaskGroup
from collections.abc import Sequence
from typing import Any

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios.suite import Suite

from .generators.base import ScenarioGenerator
from .registry import _normalize_generator


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
            child_rngs = rng.spawn(len(generators))
            for generator, n, child_rng in zip(generators, counts, child_rngs):
                tasks.append(
                    task_group.create_task(
                        generator.generate_scenario(
                            description, languages, int(n), child_rng
                        )
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
    generators: Sequence[ScenarioGenerator | type[ScenarioGenerator]],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> Suite[Any, Any]:
    """Generate a test suite by running the supplied generators.

    This generic suite builder resolves generator classes or instances,
    distributes the optional scenario budget, runs generation concurrently,
    and wraps the results in a named Suite.

    Args:
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        generators: Sequence of generator instances or classes to use.
        max_scenarios: Total upper bound on scenarios across all generators.
            None lets each generator apply its own default.
        seed: Integer seed for the top-level RNG, ensuring reproducibility
            across runs with the same arguments.

    Returns:
        A Suite containing all generated scenarios, ready for execution.
    """
    if max_scenarios is not None and max_scenarios < 0:
        raise ValueError(f"max_scenarios must be non-negative, got {max_scenarios}")

    resolved = [_normalize_generator(g) for g in generators]
    scenarios = await _generate_scenarios(
        description, languages, resolved, max_scenarios, seed
    )
    return Suite(name="Scenarios", scenarios=scenarios)
