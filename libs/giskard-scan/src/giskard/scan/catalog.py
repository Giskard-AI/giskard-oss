from asyncio import TaskGroup
from collections.abc import Sequence
from typing import Any

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.checks.scenarios.suite import Suite

from .generators.base import ScenarioContext, ScenarioGenerator, TargetMode
from .registry import _normalize_generator
from .utils.knowledge_base import KnowledgeBase, normalize_knowledge_base


async def _generate_scenarios(
    context: ScenarioContext,
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
    target_mode: TargetMode = "multiturn",
) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    rng = np.random.default_rng(seed)

    # Split the optional budget before spawning so child seeds stay stable: a
    # shared rng would be drawn from in scheduling-dependent order under
    # TaskGroup, breaking reproducibility despite the seed.
    if max_scenarios is not None and len(generators) > 0:
        counts = [
            int(n)
            for n in rng.multinomial(
                max_scenarios, np.ones(len(generators)) / len(generators)
            )
        ]
    else:
        counts = [None] * len(generators)
    child_rngs = rng.spawn(len(generators))

    tasks = []
    async with TaskGroup() as task_group:
        for generator, n, child_rng in zip(generators, counts, child_rngs):
            tasks.append(
                task_group.create_task(
                    generator.generate_scenario(
                        context,
                        max_scenarios=n,
                        rng=child_rng,
                        target_mode=target_mode,
                    )
                )
            )

    return [scenario for task in tasks for scenario in task.result()]


async def generate_suite(
    description: str,
    languages: list[str],
    generators: Sequence[ScenarioGenerator | type[ScenarioGenerator]],
    max_scenarios: int | None = None,
    seed: int = 42,
    target_mode: TargetMode = "multiturn",
    knowledge_base: KnowledgeBase | list[str] | None = None,
) -> Suite[Any, Any]:
    """Generate a test suite by running the supplied generators.

    Resolves generator classes or instances, builds one run-wide
    :class:`ScenarioContext`, distributes the optional scenario budget, runs
    generation concurrently, and wraps the results in a named Suite.

    Args:
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        generators: Sequence of generator instances or classes to use.
        max_scenarios: Total upper bound on scenarios across all generators.
            None lets each generator apply its own default.
        seed: Integer seed for the top-level RNG, ensuring reproducibility
            across runs with the same arguments.
        target_mode: Whether the agent under test supports single-turn or
            multi-turn conversations. ``"singleturn"`` skips generators that
            are multi-turn by design and caps turn budgets to 1 on others.
            Defaults to ``"multiturn"``.
        knowledge_base: Optional documents forwarded via the context to
            generators that use knowledge-base context. Raw strings are
            normalized to a :class:`KnowledgeBase`.

    Returns:
        A Suite containing all generated scenarios, ready for execution.
    """
    if max_scenarios is not None and max_scenarios < 0:
        raise ValueError(f"max_scenarios must be non-negative, got {max_scenarios}")

    context = ScenarioContext(
        description=description,
        languages=languages,
        # generate_suite is public and accepts raw list[str]; normalize once here
        # so every generator receives a KnowledgeBase | None, never list[str].
        knowledge_base=normalize_knowledge_base(knowledge_base),
    )
    resolved = [_normalize_generator(g) for g in generators]
    scenarios = await _generate_scenarios(
        context, resolved, max_scenarios, seed, target_mode=target_mode
    )
    return Suite(name="Scenarios", scenarios=scenarios)
