import logging
import sys
from asyncio import TaskGroup
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
from giskard.checks import Scenario, Suite, Trace
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
)

from .generators.base import ScenarioContext, ScenarioGenerator, TargetMode
from .registry import _normalize_generator
from .utils.knowledge_base import KnowledgeBase, normalize_knowledge_base

logger = logging.getLogger(__name__)


def _generator_name(generator: ScenarioGenerator) -> str:
    return type(generator).__name__


class _GenerationProgressReporter:
    """Callbacks for generation progress updates."""

    def __init__(
        self,
        generator_count: int,
        max_scenarios: int | None,
        use_logs: bool,
    ) -> None:
        self._generator_count = generator_count
        self._max_scenarios = max_scenarios
        self._use_logs = use_logs
        self._completed = 0
        self._scenario_count = 0
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def start_rich(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def start_logs(self) -> None:
        logger.info(
            "generate_suite: start generators=%d max_scenarios=%s",
            self._generator_count,
            self._max_scenarios,
        )

    def on_generator_done(self, generator_name: str, scenario_count: int) -> None:
        self._completed += 1
        self._scenario_count += scenario_count
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                advance=1,
                description=(
                    f"Generating suite [{self._completed}/{self._generator_count}] "
                    f"{generator_name}"
                ),
            )
        elif self._use_logs:
            logger.info(
                "generate_suite: done generator=%s scenarios=%d",
                generator_name,
                scenario_count,
            )

    def finish(self, use_rich: bool) -> None:
        if self._use_logs:
            logger.info(
                "generate_suite: finished total_scenarios=%d",
                self._scenario_count,
            )
        if use_rich and self._generator_count > 0:
            Console(stderr=True).print(f"Generated {self._scenario_count} scenarios.")


@contextmanager
def _generation_progress(
    generator_count: int,
    max_scenarios: int | None,
    verbose: bool,
) -> Iterator[_GenerationProgressReporter]:
    use_rich = verbose and sys.stderr.isatty()
    use_logs = verbose and not sys.stderr.isatty()
    reporter = _GenerationProgressReporter(generator_count, max_scenarios, use_logs)

    if use_rich and generator_count > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=Console(stderr=True),
        ) as progress:
            task_id = progress.add_task("Generating suite", total=generator_count)
            reporter.start_rich(progress, task_id)
            try:
                yield reporter
            finally:
                reporter.finish(use_rich=True)
    else:
        if use_logs:
            reporter.start_logs()
        try:
            yield reporter
        finally:
            reporter.finish(use_rich=False)


async def _generate_one(
    generator: ScenarioGenerator,
    context: ScenarioContext,
    max_scenarios: int | None,
    rng: np.random.Generator,
    target_mode: TargetMode,
    progress: _GenerationProgressReporter | None,
) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
    scenarios = await generator.generate_scenario(
        context,
        max_scenarios=max_scenarios,
        rng=rng,
        target_mode=target_mode,
    )
    if progress is not None:
        progress.on_generator_done(_generator_name(generator), len(scenarios))
    return scenarios


async def _generate_scenarios(
    context: ScenarioContext,
    generators: list[ScenarioGenerator],
    max_scenarios: int | None = None,
    seed: int = 42,
    target_mode: TargetMode = "multiturn",
    verbose: bool = True,
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

    progress_ctx = (
        _generation_progress(len(generators), max_scenarios, verbose)
        if verbose and len(generators) > 0
        else nullcontext()
    )

    with progress_ctx as progress:
        tasks = []
        async with TaskGroup() as task_group:
            for generator, n, child_rng in zip(generators, counts, child_rngs):
                tasks.append(
                    task_group.create_task(
                        _generate_one(
                            generator,
                            context,
                            n,
                            child_rng,
                            target_mode,
                            progress,
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
    verbose: bool = True,
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
        verbose: When ``True`` (default), show generation progress on a TTY
            or emit INFO logs when stderr is not a TTY. Set to ``False`` for
            fully quiet generation.

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
        context,
        resolved,
        max_scenarios,
        seed,
        target_mode=target_mode,
        verbose=verbose,
    )
    return Suite(name="Scenarios", scenarios=scenarios)
