"""Shared error mapping for httpx-based provider SDKs (OpenAI, Anthropic).

Both SDKs expose the same exception hierarchy (``RateLimitError``,
``AuthenticationError``, ``PermissionDeniedError``, ``BadRequestError``,
``APITimeoutError``, ``InternalServerError``, ``APIStatusError``,
``APIError``), so the mapping lives here once instead of once per provider.
"""

# pyright: reportAttributeAccessIssue=false

from typing import Any, NoReturn

from ..errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    RateLimitError,
    ServerError,
)


def map_httpx_sdk_error(e: Exception, sdk: Any, provider: str) -> NoReturn:
    """Map an httpx-based SDK exception to the giskard error hierarchy.

    Parameters
    ----------
    e : Exception
        The SDK exception to map.
    sdk : module
        The provider SDK module (``openai`` or ``anthropic``); its error
        classes are read off it.
    provider : str
        Provider name for the mapped error.

    Notes
    -----
    ``PermissionDeniedError`` (403) is checked before the generic
    ``APIStatusError``/``APIError`` branches: it is not a subclass of the
    SDK ``AuthenticationError``, so without this branch a 403 fell through
    to generic ``LLMError`` even though the docs promise
    ``AuthenticationError`` covers 401/403.
    """
    if isinstance(e, sdk.RateLimitError):
        raise RateLimitError(429, str(e), provider) from e
    if isinstance(e, sdk.AuthenticationError):
        raise AuthenticationError(e.status_code, str(e), provider) from e
    if isinstance(e, sdk.PermissionDeniedError):
        raise AuthenticationError(e.status_code, str(e), provider) from e
    if isinstance(e, sdk.BadRequestError):
        raise BadRequestError(e.status_code, str(e), provider) from e
    if isinstance(e, sdk.APITimeoutError):
        raise LLMTimeoutError(408, str(e), provider) from e
    if isinstance(e, sdk.InternalServerError):
        raise ServerError(e.status_code, str(e), provider) from e
    if isinstance(e, sdk.APIStatusError):
        raise LLMError(e.status_code, str(e), provider) from e
    if isinstance(e, sdk.APIError):
        raise LLMError(getattr(e, "status_code", None) or 500, str(e), provider) from e
    raise
