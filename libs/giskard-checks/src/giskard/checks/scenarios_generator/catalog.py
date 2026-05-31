from asyncio import TaskGroup
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.interaction import Trace
from ..core.scenario import Scenario
from ..scenarios.suite import Suite
from .base import ScenarioGenerator
from .registry import _normalize_generator, suite_generator_registry


async def _generate_scenarios(
    description: str,
    languages: list[str],
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    """Run all *generators* concurrently and collect their scenarios.

    When *max_scenarios* is set, the budget is split across generators via
    ``rng.multinomial``.  Each generator receives an independent child RNG
    (via ``rng.spawn``) so their internal sampling streams are statistically
    independent but the whole run is reproducible from *seed*.

    When *max_scenarios* is ``None`` every generator runs without a budget
    and child RNGs are not created.
    """
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
    generators: Sequence[ScenarioGenerator | type[ScenarioGenerator]] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
) -> Suite[Any, Any]:
    """Generate a test suite by running all registered (or supplied) generators.

    This is the primary public entry point for suite generation.  It resolves
    generators, distributes the optional scenario budget, runs generation
    concurrently, and wraps the results in a named
    :class:`~giskard.checks.scenarios.Suite`.

    Args:
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        generators: Sequence of generator instances or classes to use.
            When ``None``, all generators registered in
            :data:`~giskard.checks.scenarios_generator.registry.suite_generator_registry`
            are used.
        max_scenarios: Total upper bound on scenarios across all generators.
            ``None`` lets each generator apply its own default.
        seed: Integer seed for the top-level RNG, ensuring reproducibility
            across runs with the same arguments.

    Returns:
        A :class:`~giskard.checks.scenarios.Suite` containing all generated
        scenarios, ready for execution.
    """
    if max_scenarios is not None and max_scenarios < 0:
        raise ValueError(f"max_scenarios must be non-negative, got {max_scenarios}")

    resolved = (
        [_normalize_generator(g) for g in generators]
        if generators is not None
        else suite_generator_registry.generators()
    )
    scenarios = await _generate_scenarios(
        description, languages, resolved, max_scenarios, seed
    )
    return Suite(name="Scenarios", scenarios=scenarios)
