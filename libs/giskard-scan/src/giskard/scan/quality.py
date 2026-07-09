"""Quality scan entry points for giskard.scan."""

import logging
import time
import warnings

from giskard.checks import SuiteResult, Target, Trace
from giskard.core import telemetry_capture, telemetry_run_context, telemetry_tag

from ._telemetry_props import generator_type_counts, suite_scan_shape_properties
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

    generators = quality_suite_generator_registry.generators()
    with telemetry_run_context():
        telemetry_tag("giskard_component", "scan_quality")
        telemetry_tag("giskard_operation", "quality_scan")

        suite = await generate_suite(
            description=description,
            languages=languages,
            generators=generators,
            max_scenarios=max_scenarios,
            seed=seed,
            target_mode=target_mode,
            knowledge_base=knowledge_base,
        )

        shape_props = suite_scan_shape_properties(
            scan_kind="quality",
            language_count=len(languages),
            target_mode=target_mode,
            generator_count=len(generators),
            scenario_count=len(suite.scenarios),
            generator_types=generator_type_counts(generators),
            parallel=parallel,
            max_concurrency=max_concurrency,
            has_knowledge_base=knowledge_base is not None,
        )
        telemetry_capture("scan_quality_run_started", properties=shape_props)

        start_time = time.perf_counter()
        result: SuiteResult = await suite.run(
            target,
            parallel=parallel,
            max_concurrency=max_concurrency,
        )
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        try:
            recommendation = await generate_quality_recommendation(result)
        except Exception:
            logger.exception("Quality recommendation generation failed")
            recommendation = ""
        quality_result = result.model_copy(update={"recommendation": recommendation})

        telemetry_capture(
            "scan_quality_run_finished",
            properties={
                **shape_props,
                "duration_ms": duration_ms,
                "passed_count": quality_result.passed_count,
                "failed_count": quality_result.failed_count,
                "errored_count": quality_result.errored_count,
                "skipped_count": quality_result.skipped_count,
                "has_recommendation": bool(recommendation),
            },
        )

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
