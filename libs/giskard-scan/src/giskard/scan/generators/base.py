import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, override

import numpy as np
from giskard.checks import BaseLLMGenerator, Interact, Scenario, Trace
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..utils.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SCENARIOS = 20


class ScenarioContext(BaseModel):
    """Run-wide context shared by all generators in a scan suite.

    Carries everything every generator receives identically for a single
    run. Per-generator values (scenario budget, RNG) are passed as separate
    parameters to ``generate_scenario`` and are not part of this object.

    Attributes:
        description: Natural-language description of the agent under test.
        languages: BCP-47 language codes the agent is expected to handle.
        knowledge_base: Optional document context, ``None`` when the run has
            no knowledge base. Generators that do not use it ignore it.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    description: str
    languages: list[str]
    knowledge_base: KnowledgeBase | None = None


TargetMode = Literal["singleturn", "multiturn"]


class ScenarioGenerator(BaseModel):
    """Abstract base class for all scenario generators.

    Subclasses must implement :meth:`generate_scenario`.
    """

    @property
    def allow_commercial_use(self) -> bool:
        """Whether the generator allows commercial use.

        Returns:
            True if the generator allows commercial use, False otherwise.
        """
        return True

    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Generate a list of test scenarios for the described agent.

        Args:
            context: Run-wide :class:`ScenarioContext` describing the agent,
                its languages, and optional shared knowledge base.
            max_scenarios: Upper bound on the number of scenarios to return.
                ``None`` means no limit (generator-specific default applies).
            rng: Seeded NumPy random generator for reproducible sampling.
                In a multi-generator context, each generator receives an
                independent child RNG spawned from a shared parent via
                ``rng.spawn()``.
            target_mode: Desired conversation mode for generated scenarios.
                ``"singleturn"`` generates single-turn test cases. ``"multiturn"``
                (default) generates multi-turn test cases.

        Returns:
            A list of :class:`~giskard.checks.core.scenario.Scenario` objects
            ready to be collected into a :class:`~giskard.checks.scenarios.Suite`.
        """
        raise NotImplementedError

    @staticmethod
    def _effective_max_turns(max_turns: int, target_mode: TargetMode) -> int:
        """Resolve the per-scenario turn budget for ``target_mode``.

        ``"singleturn"`` collapses any configured ``max_turns`` to ``1``;
        ``"multiturn"`` leaves it unchanged.
        """
        return 1 if target_mode == "singleturn" else max_turns

    def _skip_for_singleturn(self, target_mode: TargetMode) -> bool:
        """Whether this multi-turn-only generator should skip ``target_mode``.

        Returns ``True`` and logs a warning when ``target_mode="singleturn"``,
        signalling the caller to return no scenarios. Generators that are
        multi-turn by design call this first in :meth:`generate_scenario`.
        """
        if target_mode == "singleturn":
            logger.warning(
                "%s requires multiturn mode; skipping (target_mode='singleturn').",
                type(self).__name__,
            )
            return True
        return False


_DATA_DIR = Path(__file__).parent / "data"


class BaseDatasetScenarioGenerator(ScenarioGenerator):
    """Base class for dataset scenario generators.

    Subclasses must implement :meth:`load_scenarios`.

    Attributes:
        tags: Tags applied to every loaded scenario via
            :meth:`~giskard.checks.core.scenario.Scenario.with_tags`.
    """

    tags: list[str] = Field(default_factory=list)

    def load_scenarios(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Load scenarios, annotating each with ``description`` and ``languages``.

        Returns:
            A list of scenarios.
        """
        raise NotImplementedError

    def _parse_scenarios(
        self,
        lines: Iterable[str],
        description: str,
        languages: list[str],
        source: str,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Parse JSONL lines into annotated scenarios.

        Shared by the bundled and Hugging Face dataset generators so the
        parsing, annotation, and tagging behaviour stays identical.

        Args:
            lines: Iterable of raw JSONL lines (blank lines are skipped).
            description: Forwarded into each scenario's annotations.
            languages: Forwarded into each scenario's annotations.
            source: Human-readable origin (path or repo file) used in error messages.

        Returns:
            A list of annotated scenarios.
        """
        scenarios: list[Scenario[Any, Any, Trace[Any, Any]]] = []
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                scenario = Scenario.model_validate_json(line)
            except ValidationError as e:
                raise ValueError(f"Malformed JSON in {source}:{line_num}: {e}") from e
            scenario = scenario.with_annotations(
                {
                    **scenario.annotations,
                    "description": description,
                    "languages": languages,
                }
            )
            if self.tags:
                scenario = scenario.with_tags(self.tags)
            scenarios.append(scenario)
        return scenarios

    @override
    async def generate_scenario(
        self,
        context: ScenarioContext,
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
        target_mode: TargetMode = "multiturn",
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Load and optionally subsample scenarios from the bundled dataset.

        Args:
            context: Run-wide context providing description and languages
                forwarded to each scenario's annotations.
            max_scenarios: Maximum number of scenarios to return.  When
                ``None``, defaults to :data:`_DEFAULT_MAX_SCENARIOS` (20).
            rng: Random generator used for subset sampling.  A fresh
                ``np.random.default_rng()`` is created if ``None``.
            target_mode: Desired conversation mode for generated scenarios.
                ``"singleturn"`` caps each interaction generator to a single
                step; ``"multiturn"`` (default) keeps the dataset's own turn
                budgets.

        Returns:
            A list of annotated :class:`~giskard.checks.core.scenario.Scenario`
            objects, ordered by their original dataset position.
        """
        # load_scenarios does blocking I/O (file reads, and network for the HF
        # subclass). Generators run concurrently via asyncio.TaskGroup, so offload
        # to a thread to avoid stalling the event loop and the other generators.
        scenarios = await asyncio.to_thread(
            self.load_scenarios, context.description, context.languages
        )

        max_scenarios = (
            max_scenarios if max_scenarios is not None else _DEFAULT_MAX_SCENARIOS
        )

        if max_scenarios < len(scenarios):
            rng = rng if rng is not None else np.random.default_rng()
            indices = rng.choice(len(scenarios), size=max_scenarios, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]

        if target_mode == "singleturn":
            for scenario in scenarios:
                self._cap_scenario_to_single_turn(scenario)

        return scenarios

    @staticmethod
    def _cap_scenario_to_single_turn(
        scenario: Scenario[Any, Any, Trace[Any, Any]],
    ) -> None:
        """Cap every interaction generator in ``scenario`` to a single step.

        Bundled dataset scenarios may encode multi-step interactions; in
        ``singleturn`` mode those generators are clamped to ``max_steps=1`` in
        place so the scenario cannot drive a multi-turn conversation against an
        agent declared single-turn.
        """
        for step in scenario.steps:
            for interact in step.interacts:
                if isinstance(interact, Interact) and isinstance(
                    interact.inputs, BaseLLMGenerator
                ):
                    interact.inputs.max_steps = 1


class LocalDatasetScenarioGenerator(BaseDatasetScenarioGenerator):
    """Scenario generator backed by a static JSONL dataset bundled in the package.

    Reads scenarios from ``<data_dir>/<dataset_name>.jsonl``, one JSON object
    per line, and annotates each with the caller-supplied ``description`` and
    ``languages``.  When ``max_scenarios`` is set and smaller than the dataset
    size, a random subset is drawn without replacement using ``rng``.

    Attributes:
        dataset_name: Stem of the ``.jsonl`` file inside the package
            ``data/`` directory (e.g. ``"prompt_injection"``).
    """

    dataset_name: str

    @override
    def load_scenarios(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        path = _DATA_DIR / f"{self.dataset_name}.jsonl"

        if not path.exists():
            raise RuntimeError(
                f"Dataset file not found: {path}. This may indicate a broken installation — try reinstalling the package."
            )

        with path.open(encoding="utf-8") as f:
            return self._parse_scenarios(
                f,
                description=description,
                languages=languages,
                source=str(path),
            )
