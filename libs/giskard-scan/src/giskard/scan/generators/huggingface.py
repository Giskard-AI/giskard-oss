import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, override

import httpx
from giskard.checks import Scenario, Trace
from huggingface_hub import DatasetCard, hf_hub_download, list_repo_files
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)
from pydantic import Field

from .base import BaseDatasetScenarioGenerator

logger = logging.getLogger(__name__)

_HUB_UNAVAILABLE_ERRORS = (
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    httpx.NetworkError,
    httpx.ProxyError,
)
_HUB_LOAD_ERRORS = _HUB_UNAVAILABLE_ERRORS + (HfHubHTTPError,)


def _hub_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, _HUB_UNAVAILABLE_ERRORS):
        return True
    if isinstance(exc, HfHubHTTPError):
        return exc.response.status_code in (502, 503, 504)
    return False


def _handle_hub_outage(repo_id: str, exc: BaseException) -> bool:
    """Log and swallow hub-outage errors. Returns True when handled."""
    if not _hub_unavailable(exc):
        return False
    logger.warning(
        "Hugging Face Hub is unavailable for %s (%s); returning no scenarios.",
        repo_id,
        exc,
    )
    return True


def _resolve_data_files(data_files: Any) -> list[str]:
    """Return string ``path`` values from a config's ``data_files`` list."""
    if not data_files:
        return []
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list):
        paths: list[str] = []
        for entry in data_files:
            if isinstance(entry, str):
                paths.append(entry)
            elif isinstance(entry, dict):
                path = entry.get("path")
                if isinstance(path, str):
                    paths.append(path)
        return paths
    return []


@lru_cache(maxsize=32)
def _language_subsets(repo_id: str) -> dict[str, list[str]]:
    """Map each config name to repo files present in the dataset (cached per repo)."""
    card = DatasetCard.load(repo_id, repo_type="dataset")
    card_data = getattr(card, "data", None)
    configs = getattr(card_data, "configs", None) or [] if card_data is not None else []
    repo_files = set(list_repo_files(repo_id, repo_type="dataset"))

    subsets: dict[str, list[str]] = {}
    for config in configs:
        if not isinstance(config, dict):
            continue
        name = config.get("config_name")
        if not name:
            continue
        data_files = config.get("data_files")
        present = [p for p in _resolve_data_files(data_files) if p in repo_files]
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
        try:
            subsets = _language_subsets(self.repo_id)
        except _HUB_LOAD_ERRORS as exc:
            if not _handle_hub_outage(self.repo_id, exc):
                raise
            return []

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
        try:
            for language in compatible:
                for repo_file in subsets[language]:
                    local_path = hf_hub_download(
                        self.repo_id, repo_file, repo_type="dataset"
                    )
                    with Path(local_path).open(encoding="utf-8") as f:
                        scenarios.extend(
                            self._parse_scenarios(
                                f,
                                description=description,
                                languages=languages,
                                source=f"{self.repo_id}/{repo_file}",
                            )
                        )
        except _HUB_LOAD_ERRORS as exc:
            if not _handle_hub_outage(self.repo_id, exc):
                raise
            return []

        return scenarios
