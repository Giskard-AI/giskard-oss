"""Quality scan entry points for giskard.scan."""

import logging
import warnings

from giskard.checks import SuiteResult, Target, Trace

from .catalog import generate_suite
from .generators.base import TargetMode
from .generators.knowledge_base import (
    HallucinationScenarioGenerator,
    KnowledgeBaseScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    SycophancyScenarioGenerator,
)
from .registry import SuiteGeneratorRegistry
from .utils.knowledge_base import KnowledgeBase, normalize_knowledge_base
from .utils.recommendation import generate_quality_recommendation

logger = logging.getLogger(__name__)

QUALITY_GENERATOR_TYPES: tuple[type[KnowledgeBaseScenarioGenerator], ...] = (
    HallucinationScenarioGenerator,
    SycophancyScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
)

quality_suite_generator_registry = SuiteGeneratorRegistry()
for generator_type in QUALITY_GENERATOR_TYPES:
    quality_suite_generator_registry.register(generator_type)


async def quality_scan[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    target: Target[InputType, OutputType, TraceType],
    description: str,
    languages: list[str],
    knowledge_base: KnowledgeBase | list[str] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
    group_by: str | None = "component",
    parallel: bool = True,
    max_concurrency: int | None = None,
    return_exception: bool = False,
    target_mode: TargetMode = "multiturn",
) -> SuiteResult:
    """Generate and run the standard quality scan suite.

    Builds a suite from the quality scenario generator registry, runs it
    against the target, prints the grouped report with a recommendation, and
    returns the suite result.

    Args:
        target: Agent or provider target to evaluate.
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        knowledge_base: Documents used by knowledge-base quality generators.
            Raw strings are converted to a :class:`KnowledgeBase`.
        max_scenarios: Total upper bound on scenarios across all quality
            generators. ``None`` lets each generator apply its own default.
        seed: Integer seed used for reproducible scenario generation.
        group_by: Result annotation key used to group the printed report.
            ``None`` prints the ungrouped report.
        parallel: When ``True``, run scenarios concurrently (default).
        max_concurrency: Cap on concurrent scenarios when ``parallel=True``.
            ``None`` runs all scenarios at once.
        return_exception: When ``True``, a scenario whose input generation fails
            is recorded as an errored result and the scan continues. When
            ``False`` (default), the failure propagates and aborts the scan.
        target_mode: Whether the agent under test supports single-turn or
            multi-turn conversations. ``"singleturn"`` skips generators that
            are multi-turn by design and caps turn budgets to 1 on others.
            Defaults to ``"multiturn"``.

    Returns:
        The completed suite result with a generated quality recommendation.
    """
    knowledge_base = normalize_knowledge_base(
        _warn_if_missing_knowledge_base(knowledge_base)
    )

    suite = await generate_suite(
        description=description,
        languages=languages,
        generators=quality_suite_generator_registry.generators(),
        max_scenarios=max_scenarios,
        seed=seed,
        target_mode=target_mode,
        knowledge_base=knowledge_base,
    )

    result: SuiteResult = await suite.run(
        target,
        parallel=parallel,
        max_concurrency=max_concurrency,
        return_exception=return_exception,
    )
    try:
        recommendation = await generate_quality_recommendation(result)
    except Exception:
        logger.exception("Quality recommendation generation failed")
        recommendation = ""
    quality_result = result.model_copy(update={"recommendation": recommendation})
    quality_result.print_report(group_by=group_by)
    return quality_result


def _warn_if_missing_knowledge_base(
    knowledge_base: KnowledgeBase | list[str] | None,
) -> KnowledgeBase | list[str] | None:
    if knowledge_base is None:
        warnings.warn(
            "quality_scan received no knowledge base; knowledge-base quality scenarios will be skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if isinstance(knowledge_base, KnowledgeBase):
        if knowledge_base.documents:
            return knowledge_base
    elif any(document.strip() for document in knowledge_base):
        return knowledge_base

    warnings.warn(
        "quality_scan received an empty knowledge base; knowledge-base quality scenarios will be skipped.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None
