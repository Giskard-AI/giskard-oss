"""Quality scan entry points for giskard.scan."""

import logging
import warnings

from giskard.checks import SuiteResult, Target, Trace

from .catalog import generate_suite
from .generators.base import DEFAULT_TARGET_MODE, TargetMode
from .generators.knowledge_base import (
    HallucinationScenarioGenerator,
    KnowledgeBaseScenarioGenerator,
    MultiTopicScenarioGenerator,
    OutOfScopeScenarioGenerator,
    SplitQuestionsScenarioGenerator,
    SycophancyScenarioGenerator,
)
from .registry import SuiteGeneratorRegistry
from .types import resolve_scan_options
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
    *,
    knowledge_base: KnowledgeBase | list[str] | None = None,
    max_scenarios: int | None = None,
    seed: int = 42,
    group_by: str | None = "component",
    parallel: bool = True,
    max_concurrency: int | None = None,
    return_exception: bool = False,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
) -> SuiteResult:
    """Generate and run the standard quality scan suite.

    Builds a suite from the quality scenario generator registry, runs it against
    the target, prints the grouped report with a recommendation, and returns
    the suite result.

    Parameters
    ----------
    target : Target
        Agent or provider target to evaluate.
    description : str
        Natural-language description of the agent under test.
    languages : list of str
        BCP-47 language codes the agent is expected to handle.
    knowledge_base : KnowledgeBase, list of str, or None, optional
        Documents used by knowledge-base quality generators. Raw strings are
        converted to a :class:`~giskard.scan.utils.knowledge_base.KnowledgeBase`.
    max_scenarios : int, optional
        Total upper bound on scenarios across all quality generators. ``None``
        lets each generator apply its own default.
    seed : int, optional
        Integer seed used for reproducible scenario generation. Defaults to ``42``.
    group_by : str, optional
        Result tag key used to group the printed report. ``None`` prints
        the ungrouped report. Defaults to ``"component"``.
    parallel : bool, optional
        When ``True`` (default), run generated scenarios concurrently against
        the target. Pass ``False`` for serial execution. This is suite
        *execution*; scenario *generation* always runs generators concurrently
        via :func:`~giskard.scan.catalog.generate_suite`.
    max_concurrency : int, optional
        Cap on concurrent scenarios when ``parallel=True``. ``None`` runs all
        scenarios at once (provider rate limits become the effective cap).
        When ``parallel=False``, a valid value has no effect on scheduling,
        but invalid values are still rejected.
    return_exception : bool, optional
        When ``True``, a scenario whose input generation fails is recorded as an
        errored result and the scan continues. When ``False`` (default), the
        failure propagates and aborts the scan.
    target_mode : {"singleturn", "multiturn"}, optional
        Whether the agent under test supports single-turn or multi-turn
        conversations. ``"singleturn"`` skips generators that are multi-turn by
        design and caps turn budgets to 1 on others. Defaults to
        :data:`~giskard.scan.generators.base.DEFAULT_TARGET_MODE`.

    Returns
    -------
    SuiteResult
        The completed suite result with a generated quality recommendation.
    """
    opts = resolve_scan_options(
        max_scenarios=max_scenarios,
        seed=seed,
        group_by=group_by,
        parallel=parallel,
        max_concurrency=max_concurrency,
        return_exception=return_exception,
    )

    knowledge_base = normalize_knowledge_base(
        _warn_if_missing_knowledge_base(knowledge_base)
    )

    suite = await generate_suite(
        description=description,
        languages=languages,
        generators=quality_suite_generator_registry.generators(),
        max_scenarios=opts["max_scenarios"],
        seed=opts["seed"],
        target_mode=target_mode,
        knowledge_base=knowledge_base,
    )

    result: SuiteResult = await suite.run(
        target,
        parallel=opts["parallel"],
        max_concurrency=opts["max_concurrency"],
        return_exception=opts["return_exception"],
    )
    try:
        recommendation = await generate_quality_recommendation(result)
    except Exception:
        logger.exception("Quality recommendation generation failed")
        recommendation = ""
    quality_result = result.model_copy(update={"recommendation": recommendation})
    quality_result.print_report(group_by=opts["group_by"])
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
