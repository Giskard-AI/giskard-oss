"""OrcaRouter provider using the ``openai`` SDK.

Routing prefix: ``orcarouter/``

Authentication:
    - Env: ``ORCAROUTER_API_KEY``
    - Kwargs: ``api_key``, ``base_url``, ``timeout``

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
    - ``base_url``: OrcaRouter API endpoint (default: ``https://api.orcarouter.ai/v1``)
    - ``timeout``: request timeout in seconds
    - ``http_client``: caller-owned async HTTP client passed to the SDK; not closed by giskard-llm
    - ``default_headers``: extra headers merged into every SDK request
"""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .openai import OpenAIProvider

if TYPE_CHECKING:
    from httpx import AsyncClient

PROVIDER = "orcarouter"

DEFAULT_BASE_URL = "https://api.orcarouter.ai/v1"


class OrcaRouterProvider(OpenAIProvider):
    _PROVIDER = "orcarouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: "AsyncClient | None" = None,
        default_headers: Mapping[str, str] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("ORCAROUTER_API_KEY"),
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
            http_client=http_client,
            default_headers=default_headers,
            **_kwargs,
        )
