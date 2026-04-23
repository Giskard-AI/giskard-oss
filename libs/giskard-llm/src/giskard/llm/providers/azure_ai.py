"""Azure AI Foundry provider using the ``openai`` SDK (OpenAI-compatible endpoint).

Routing prefix: ``azure_ai/``

Authentication:
    - Env: ``AZURE_AI_API_KEY``, ``AZURE_AI_ENDPOINT``
    - Kwargs: ``api_key``, ``base_url``

Role mapping:
    Same as OpenAI — all canonical roles passed through as-is. Azure AI
    Foundry endpoints expose an OpenAI-compatible API.

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

Provider-specific kwargs:
    - ``base_url``: Azure AI Foundry endpoint URL. A bare Foundry root URL
      (``https://<resource>.services.ai.azure.com``) is auto-suffixed with
      ``/models`` to reach the Azure AI Model Inference API, matching how
      litellm's ``azure_ai/`` provider shaped the URL. URLs that already
      include a path are passed through unchanged.
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
_FOUNDRY_DEFAULT_PATH = "/models"


def _shape_foundry_base_url(base_url: str | None) -> str | None:
    """Append ``/models`` to a bare Azure AI Foundry root URL.

    Foundry hosts (``*.services.ai.azure.com``) serve the OpenAI-compatible
    inference surface under ``/models``, so handing a bare Foundry root URL
    to ``openai.AsyncOpenAI`` produces 404 on ``/chat/completions``. If the
    caller already specified a path (e.g. ``/openai/v1`` or ``/models/...``),
    the URL is returned unchanged. Non-Foundry hosts are always untouched.

    Mirrors litellm's ``azure_ai`` path-shaping so an ``AZURE_AI_API_BASE``
    that previously worked with litellm continues to work here.
    """
    if not base_url:
        return base_url
    parsed = urlparse(base_url)
    if not parsed.netloc.endswith(_FOUNDRY_HOST_SUFFIX):
        return base_url
    if parsed.path and parsed.path != "/":
        return base_url
    return urlunparse(parsed._replace(path=_FOUNDRY_DEFAULT_PATH))


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
        resolved_base = _shape_foundry_base_url(
            base_url or os.environ.get("AZURE_AI_ENDPOINT")
        )

        self._client = openai.AsyncOpenAI(
            **compact(api_key=resolved_key, base_url=resolved_base, timeout=timeout)
        )
