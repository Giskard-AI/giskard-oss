import logging
from typing import Any, override

from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from giskard.scan.generators.base import BaseDatasetScenarioGenerator
from huggingface_hub import hf_hub_download, list_repo_files
from pydantic import Field

logger = logging.getLogger(__name__)

_LANGUAGE_PLACEHOLDER = "{language}"


class HuggingFaceDatasetScenarioGenerator(BaseDatasetScenarioGenerator):
    """Scenario generator backed by a Hugging Face dataset.

    Loads scenarios from a Hugging Face dataset repository and annotates them
    with the caller-supplied ``description`` and ``languages``.

    The ``filename`` is a template containing a ``{language}`` placeholder
    (e.g. ``"donotanswer.{language}.jsonl"``). Available languages are
    discovered by listing the repo files and matching them against the
    template. For each requested language that the repo provides, the matching
    file is downloaded and its scenarios are concatenated. Requested languages
    with no matching file are skipped; if none match, an empty list is returned
    and a warning is emitted.

    Attributes:
        repo_id: Hugging Face dataset repository id (e.g. ``"giskardai/do-not-answer-scenarios"``).
        filename: File template with a ``{language}`` placeholder.
        repo_allow_commercial_use: Whether the dataset's license permits
            commercial use. Set explicitly per repo (the license recorded on
            the Hub card is not always authoritative).
    """

    repo_id: str
    filename: str
    repo_allow_commercial_use: bool = Field(default=True)

    @property
    @override
    def allow_commercial_use(self) -> bool:
        return self.repo_allow_commercial_use

    def _available_languages(self) -> dict[str, str]:
        """Map available language code -> repo filename via the template."""
        prefix, _, suffix = self.filename.partition(_LANGUAGE_PLACEHOLDER)
        available: dict[str, str] = {}
        for repo_file in list_repo_files(self.repo_id, repo_type="dataset"):
            if repo_file.startswith(prefix) and repo_file.endswith(suffix):
                language = repo_file[len(prefix) : len(repo_file) - len(suffix)]
                if language:
                    available[language] = repo_file
        return available

    @override
    def load_scenarios(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        available = self._available_languages()
        compatible = [language for language in languages if language in available]

        if not compatible:
            logger.warning(
                "No compatible language found in %s for requested languages %s "
                "(available: %s); returning no scenarios.",
                self.repo_id,
                languages,
                sorted(available),
            )
            return []

        scenarios: list[Scenario[Any, Any, Trace[Any, Any]]] = []
        for language in compatible:
            repo_file = available[language]
            local_path = hf_hub_download(self.repo_id, repo_file, repo_type="dataset")
            with open(local_path, encoding="utf-8") as f:
                scenarios.extend(
                    self._parse_scenarios(
                        f,
                        description=description,
                        languages=languages,
                        source=f"{self.repo_id}/{repo_file}",
                    )
                )

        return scenarios
