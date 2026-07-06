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
    ~270 of Eden AI's models are EU-hosted. ``GET /v3/models`` exposes a
    ``regions`` list per model; pick an id whose regions include ``eu`` (some
    providers also publish explicit ``@eu`` variants, e.g.
    ``edenai/amazon/amazon.nova-2-lite-v1:0@eu``) to keep inference within the
    European Union.

Authentication:
    - Env: ``EDENAI_API_KEY``
    - Kwargs: ``api_key``, ``base_url`` (defaults to ``https://api.edenai.run/v3``)

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


class EdenAIProvider(OpenAIProvider):
    _PROVIDER = "edenai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: "AsyncClient | None" = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("EDENAI_API_KEY"),
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
            http_client=http_client,
            default_headers=default_headers,
            **kwargs,
        )
