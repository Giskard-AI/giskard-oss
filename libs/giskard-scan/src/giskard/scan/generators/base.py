from collections.abc import Iterable
from pathlib import Path
from typing import Any, override

import numpy as np
from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from pydantic import BaseModel, Field, ValidationError


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
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Generate a list of test scenarios for the described agent.

        Args:
            description: Natural-language description of the agent under test.
            languages: BCP-47 language codes the agent is expected to handle.
            max_scenarios: Upper bound on the number of scenarios to return.
                ``None`` means no limit (generator-specific default applies).
            rng: Seeded NumPy random generator for reproducible sampling.
                When used in a multi-generator context, each generator receives
                an independent child RNG spawned from a shared parent via
                ``rng.spawn()``, ensuring statistical independence while
                maintaining reproducibility. Direct callers typically pass a
                fresh generator or ``None`` to let the implementation create
                one.

        Returns:
            A list of :class:`~giskard.checks.core.scenario.Scenario` objects
            ready to be collected into a :class:`~giskard.checks.scenarios.Suite`.
        """
        raise NotImplementedError


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
        description: str,
        languages: list[str],
        max_scenarios: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        """Load and optionally subsample scenarios from the bundled dataset.

        Args:
            description: Forwarded to each scenario's annotations so that
                downstream judges know which agent is under test.
            languages: Forwarded to each scenario's annotations.
            max_scenarios: Maximum number of scenarios to return.  When
                ``None``, the full dataset is returned.
            rng: Random generator used for subset sampling.  A fresh
                ``np.random.default_rng()`` is created if ``None``.

        Returns:
            A list of annotated :class:`~giskard.checks.core.scenario.Scenario`
            objects, ordered by their original dataset position.
        """
        scenarios = self.load_scenarios(description, languages)

        if max_scenarios is not None and max_scenarios < len(scenarios):
            rng = rng if rng is not None else np.random.default_rng()
            indices = rng.choice(len(scenarios), size=max_scenarios, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]

        return scenarios


class DatasetScenarioGenerator(BaseDatasetScenarioGenerator):
    """Scenario generator backed by a static JSONL dataset.

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
