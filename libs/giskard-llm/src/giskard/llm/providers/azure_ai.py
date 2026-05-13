"""Azure AI Foundry provider using the ``openai`` SDK (``AsyncAzureOpenAI``).

Routing prefix: ``azure_ai/``

Authentication:
    - Env: ``AZURE_AI_API_KEY``, ``AZURE_AI_ENDPOINT``
    - Optional env: ``AZURE_AI_API_VERSION`` (default ``2024-10-21``, aligned with ``azure/``)
    - Kwargs: ``api_key``, ``base_url``

Role mapping:
    Same as OpenAI — all canonical roles passed through as-is.

Message constraints:
    Same as OpenAI.

Tool call format:
    Same as OpenAI.

Error mapping:
    Same as OpenAI.

Supported features:
    - Completion: yes
    - Embeddings: depends on deployed model
    - Structured output (response_format): depends on deployed model

Implementation note:
    Foundry hosts (``*.services.ai.azure.com``) route chat and embeddings via
    ``/openai/deployments/{deployment-id}/…``. The SDK's ``AsyncAzureOpenAI`` uses
    that layout. A bare Foundry URL must be the **resource root** (no trailing
    ``/models`` path): posting to ``/models/embeddings`` on several Foundry tenants
    returns HTTP 200 with an empty body, which breaks plain ``AsyncOpenAI``.

Provider-specific kwargs:
    - ``base_url``: Azure AI Foundry **resource endpoint** URL (typically
      ``https://<resource>.services.ai.azure.com``). If callers still pass a URL
      that only appends legacy ``/models``, that suffix is stripped for Foundry hosts.
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportImplicitRelativeImport=false, reportMissingSuperCall=false

import logging
import os
from typing import Any
from urllib.parse import urlparse, urlunparse

from ..errors import ProviderNotAvailableError
from ..utils import compact
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

PROVIDER = "azure_ai"

_FOUNDRY_HOST_SUFFIX = ".services.ai.azure.com"
_DEFAULT_API_VERSION = "2024-10-21"


def _normalize_azure_ai_endpoint(base_url: str | None) -> str | None:
    """Normalize Foundry endpoint URLs for ``AsyncAzureOpenAI``.

    On ``*.services.ai.azure.com``, use the resource root (scheme + netloc only).
    Legacy configs that appended ``/models`` for plain ``AsyncOpenAI`` strip that
    suffix so deployments resolve to ``/openai/deployments/{model}/…``.
    """
    if base_url is None:
        return None
    raw = base_url.strip()
    if raw == "":
        return ""
    parsed = urlparse(raw)
    if not parsed.netloc.endswith(_FOUNDRY_HOST_SUFFIX):
        return raw
    path = (parsed.path or "").rstrip("/")
    if path in ("", "/models"):
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return raw


class AzureAIProvider(OpenAIProvider):
    _PROVIDER = "azure_ai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
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
            raise ProviderNotAvailableError(PROVIDER, "openai", extra="azure") from exc

        resolved_key = api_key or os.environ.get("AZURE_AI_API_KEY")
        resolved_endpoint = _normalize_azure_ai_endpoint(
            base_url or os.environ.get("AZURE_AI_ENDPOINT")
        )
        resolved_version = os.environ.get("AZURE_AI_API_VERSION", _DEFAULT_API_VERSION)

        self._client = openai.AsyncAzureOpenAI(
            **compact(
                api_key=resolved_key,
                azure_endpoint=resolved_endpoint,
                api_version=resolved_version,
                timeout=timeout,
            )
        )
