"""Eden AI provider using the ``openai`` SDK against Eden AI's v3 API.

Routing prefix: ``edenai/``

`Eden AI <https://www.edenai.co/>`_ is a unified, EU-based AI gateway exposing
500+ models from every major provider (OpenAI, Anthropic, Google, Mistral,
Amazon, xAI, DeepSeek, Groq, ...) behind a single OpenAI-compatible endpoint.
Because the endpoint is OpenAI-compatible, this provider is a thin subclass of
:class:`~giskard.llm.providers.openai.OpenAIProvider` with the base URL and
auth defaults preset.

Model names use Eden AI's ``provider/model`` form, so full giskard model
strings look like ``edenai/openai/gpt-4o`` or
``edenai/anthropic/claude-3-5-sonnet-latest`` (the router splits on the first
``/`` only, leaving ``provider/model`` as the model name).

EU data residency (GDPR):
    Pass ``eu=True`` (or set ``base_url`` to ``https://api.eu.edenai.run/v3``)
    to route through Eden AI's dedicated EU endpoint, which serves only
    EU-hosted models (~270) and rejects any non-EU model with HTTP 451,
    guaranteeing inference stays within the European Union. ``GET /v3/models``
    also exposes a ``regions`` list per model if you prefer to pick EU-hosted
    ids against the default endpoint.

Authentication:
    - Env: ``EDENAI_API_KEY``
    - Kwargs: ``api_key``, ``eu`` (use the EU endpoint), ``base_url``
      (defaults to ``https://api.edenai.run/v3``)

Supported features:
    - Completion: yes
    - Embeddings: yes
    - Responses API: yes

Role mapping, message constraints, tool format and error mapping are inherited
from :class:`~giskard.llm.providers.openai.OpenAIProvider`.
"""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .openai import OpenAIProvider

if TYPE_CHECKING:
    from httpx import AsyncClient

PROVIDER = "edenai"
DEFAULT_BASE_URL = "https://api.edenai.run/v3"
# Dedicated EU endpoint: serves only EU-hosted models and rejects any non-EU
# model with HTTP 451, guaranteeing inference stays within the European Union.
EU_BASE_URL = "https://api.eu.edenai.run/v3"


class EdenAIProvider(OpenAIProvider):
    _PROVIDER = "edenai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        eu: bool = False,
        timeout: float | None = None,
        http_client: "AsyncClient | None" = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Precedence: explicit base_url > eu=True (EU endpoint) > default endpoint.
        resolved_base_url = base_url or (EU_BASE_URL if eu else DEFAULT_BASE_URL)
        super().__init__(
            api_key=api_key or os.environ.get("EDENAI_API_KEY"),
            base_url=resolved_base_url,
            timeout=timeout,
            http_client=http_client,
            default_headers=default_headers,
            **kwargs,
        )
