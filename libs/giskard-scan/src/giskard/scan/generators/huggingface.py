import logging
from functools import lru_cache
from typing import Any, override

from giskard.checks.core.interaction import Trace
from giskard.checks.core.scenario import Scenario
from huggingface_hub import DatasetCard, hf_hub_download, list_repo_files
from pydantic import Field

from .base import BaseDatasetScenarioGenerator

logger = logging.getLogger(__name__)


def _resolve_data_files(data_files: Any) -> list[str]:
    """Flatten a config's ``data_files`` into a list of repo file paths.

    Expects the shape our datasets use: a list of ``{split, path}`` dicts with a
    string ``path``. Malformed entries (non-dict, or a missing/non-string
    ``path``) are skipped. Returns ``[]`` for ``None`` or an empty list.
    """
    if not data_files:
        return []
    paths: list[str] = []
    for entry in data_files:
        if isinstance(entry, dict):
            path = entry.get("path")
            if isinstance(path, str):
                paths.append(path)
    return paths


@lru_cache(maxsize=32)
def _language_subsets(repo_id: str) -> dict[str, list[str]]:
    """Map each subset (config) name to the repo files it resolves to.

    Reads the dataset card's ``configs`` declaration, then keeps each config's
    ``data_files`` paths that actually exist in the repo. A subset is treated as
    a language: requesting ``"en"`` loads the ``"en"`` config's files. Configs
    that resolve to no present file are dropped.

    Cached per ``repo_id``: the card and file list are static within a run, so
    this avoids hitting the Hub on every scan.
    """
    card = DatasetCard.load(repo_id, repo_type="dataset")
    configs = getattr(card.data, "configs", None) or []
    repo_files = set(list_repo_files(repo_id, repo_type="dataset"))

    subsets: dict[str, list[str]] = {}
    for config in configs:
        name = config.get("config_name")
        if not name:
            continue
        present = [
            p for p in _resolve_data_files(config.get("data_files")) if p in repo_files
        ]
        if present:
            subsets[name] = present
    return subsets


class HuggingFaceDatasetScenarioGenerator(BaseDatasetScenarioGenerator):
    """Scenario generator backed by a Hugging Face dataset.

    Loads scenarios from a Hugging Face dataset repository and annotates them
    with the caller-supplied ``description`` and ``languages``.

    The dataset must declare one *subset* (config) per language in its dataset
    card, named by the BCP-47 language code (e.g. a ``"en"`` config). Available
    languages are discovered by reading the card's ``configs`` and resolving
    each subset's ``data_files`` against the repo file list, so a language may
    span several files. For each requested language the dataset provides, the
    matching files are downloaded and their scenarios concatenated. Requested
    languages with no matching subset are skipped; if none match, an empty list
    is returned and a warning is emitted.

    Attributes:
        repo_id: Hugging Face dataset repository id (e.g. ``"giskardai/do-not-answer-scenarios"``).
        repo_allow_commercial_use: Whether the dataset's license permits
            commercial use. Set explicitly per repo (the license recorded on
            the Hub card is not always authoritative).
    """

    repo_id: str
    repo_allow_commercial_use: bool = Field(default=True)

    @property
    @override
    def allow_commercial_use(self) -> bool:
        return self.repo_allow_commercial_use

    @override
    def load_scenarios(
        self, description: str, languages: list[str]
    ) -> list[Scenario[Any, Any, Trace[Any, Any]]]:
        subsets = _language_subsets(self.repo_id)
        compatible = [language for language in languages if language in subsets]

        if not compatible:
            logger.warning(
                "No compatible language subset found in %s for requested languages "
                "%s (available: %s); returning no scenarios.",
                self.repo_id,
                languages,
                sorted(subsets),
            )
            return []

        scenarios: list[Scenario[Any, Any, Trace[Any, Any]]] = []
        for language in compatible:
            for repo_file in subsets[language]:
                local_path = hf_hub_download(
                    self.repo_id, repo_file, repo_type="dataset"
                )
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
