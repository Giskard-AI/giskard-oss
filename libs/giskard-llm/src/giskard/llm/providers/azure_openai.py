"""Azure OpenAI provider using the ``openai`` SDK with ``AsyncAzureOpenAI``.

Routing prefix: ``azure/``

Authentication:
    - Env: ``AZURE_API_KEY``, ``AZURE_API_BASE``, ``AZURE_API_VERSION``
    - Kwargs: ``api_key``, ``base_url``, ``api_version``

Role mapping:
    Same as OpenAI — all canonical roles passed through as-is.

Message constraints:
    Same as OpenAI — multiple system messages supported, no alternation.

Tool call format:
    Same as OpenAI — native format.

Error mapping:
    Same as OpenAI.

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Structured output (response_format): yes

Provider-specific kwargs:
    - ``api_version``: Azure API version (default: ``2024-10-21``)
    - ``base_url``: Azure endpoint URL
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false, reportMissingSuperCall=false

import logging
import os
from typing import Any

from ..errors import ProviderNotAvailableError
from ..utils.compact import compact
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

PROVIDER = "azure"


class AzureOpenAIProvider(OpenAIProvider):
    _PROVIDER = "azure"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        **_kwargs: Any,
    ) -> None:
        if _kwargs:
            logger.warning(
                "%s provider: ignoring unknown kwargs: %s", PROVIDER, sorted(_kwargs)
            )
        try:
            import openai
        except ImportError as exc:
            raise ProviderNotAvailableError(PROVIDER, "openai") from exc

        resolved_key = api_key or os.environ.get("AZURE_API_KEY")
        resolved_base = base_url or os.environ.get("AZURE_API_BASE")
        resolved_version = api_version or os.environ.get(
            "AZURE_API_VERSION", "2024-10-21"
        )

        self._client = openai.AsyncAzureOpenAI(
            **compact(
                api_key=resolved_key,
                azure_endpoint=resolved_base,
                api_version=resolved_version,
                timeout=timeout,
            )
        )
