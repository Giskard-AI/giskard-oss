from asyncio import TaskGroup
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from giskard.checks import (
    RunStore,
    Scenario,
    Suite,
    Trace,
    ensure_checkpoint_id,
    generate_fingerprint,
    generator_checkpoint_key,
    resolve_checkpoint_options,
)

from .generators.base import ScenarioContext, ScenarioGenerator, TargetMode
from .registry import _normalize_generator
from .utils.knowledge_base import KnowledgeBase, normalize_knowledge_base

type ResumeMode = bool | Literal["force"]


async def _generate_scenarios(
    context: ScenarioContext,
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
    target_mode: TargetMode = "multiturn",
    store: RunStore | None = None,
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

    generator_keys = [
        generator_checkpoint_key(generator, index)
        for index, generator in enumerate(generators)
    ]
    completed = store.completed_ids("generator_completed") if store else set()
    cached_payloads = store.load_payloads("scenario_generated") if store else {}

    async def run_one(
        generator: ScenarioGenerator,
        n: int | None,
        child_rng: np.random.Generator,
        gen_key: str,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        if gen_key in completed and store is not None:
            scenario_ids = store.load_payloads("generator_completed")[gen_key].get(
                "scenario_ids", []
            )
            scenarios: list[Scenario[Any, Any, Trace[Any, Any]]] = []
            for scenario_id in scenario_ids:
                payload = cached_payloads[scenario_id]
                scenarios.append(Scenario.model_validate(payload))
            return scenarios

        scenarios = await generator.generate_scenario(
            context,
            max_scenarios=n,
            rng=child_rng,
            target_mode=target_mode,
        )
        if store is not None:
            scenario_ids: list[str] = []
            for index, scenario in enumerate(scenarios):
                cid = ensure_checkpoint_id(
                    scenario.annotations,
                    name=scenario.name,
                    index=index,
                    suite_name=gen_key,
                )
                await store.append(
                    "scenario_generated",
                    id=cid,
                    payload=scenario.model_dump(mode="json", exclude={"target"}),
                )
                scenario_ids.append(cid)
            await store.append(
                "generator_completed",
                id=gen_key,
                payload={"scenario_ids": scenario_ids},
            )
        return scenarios

    tasks = []
    async with TaskGroup() as task_group:
        for generator, n, child_rng, gen_key in zip(
            generators, counts, child_rngs, generator_keys, strict=True
        ):
            tasks.append(
                task_group.create_task(run_one(generator, n, child_rng, gen_key))
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
    checkpoint_dir: Path | str | bool | None = None,
    resume: ResumeMode | None = None,
) -> Suite[Any, Any]:
    """Generate a test suite by running the supplied generators.

    Resolves generator classes or instances, builds one run-wide
    :class:`ScenarioContext`, distributes the optional scenario budget, runs
    generators concurrently (always — there is no serial generation flag),
    and wraps the results in a named Suite.

    Concurrency here is *generation* only. Whether scenarios later run against
    a target in parallel is controlled by
    :meth:`~giskard.checks.scenarios.suite.Suite.run`'s ``parallel`` argument
    (or by scan helpers that pass it through).

    Args:
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        generators: Sequence of generator instances or classes to use.
        max_scenarios: Total upper bound on scenarios across all generators.
            None lets each generator apply its own default.
        seed: Integer seed for the top-level RNG, ensuring reproducibility
            across runs with the same arguments. Child RNGs are spawned before
            concurrent generation so results stay stable under TaskGroup
            scheduling.
        target_mode: Whether the agent under test supports single-turn or
            multi-turn conversations. ``"singleturn"`` skips generators that
            are multi-turn by design and caps turn budgets to 1 on others.
            Defaults to ``"multiturn"``.
        knowledge_base: Optional documents forwarded via the context to
            generators that use knowledge-base context. Raw strings are
            normalized to a :class:`KnowledgeBase`.
        checkpoint_dir: Checkpoint root. ``None`` uses
            ``.giskard/checkpoints/<fingerprint>/``. ``False`` disables.
            A path is used as root with a fingerprint subdir.
        resume: Skip finished generators. Defaults to ``True`` when
            checkpointing is on. Use ``"force"`` to ignore fingerprint mismatches.

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

    generator_keys = [
        generator_checkpoint_key(generator, index)
        for index, generator in enumerate(resolved)
    ]
    fingerprint = generate_fingerprint(
        description=description,
        languages=languages,
        generator_keys=generator_keys,
        seed=seed,
        target_mode=target_mode,
        max_scenarios=max_scenarios,
    )
    ck_path, resume_mode = resolve_checkpoint_options(
        checkpoint_dir, resume, fingerprint=fingerprint
    )
    store: RunStore | None = None
    if ck_path is not None:
        store = await RunStore.open(
            ck_path, fingerprint=fingerprint, resume=resume_mode
        )

    scenarios = await _generate_scenarios(
        context,
        resolved,
        max_scenarios,
        seed,
        target_mode=target_mode,
        store=store,
    )
    return Suite(name="Scenarios", scenarios=scenarios)
