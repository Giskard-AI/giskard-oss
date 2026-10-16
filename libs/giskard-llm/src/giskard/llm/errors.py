"""Unified error hierarchy for giskard-llm.

Every error carries a ``status_code`` attribute so the retry middleware in
giskard-agents (which does ``getattr(err, "status_code", 0)``) works
unchanged regardless of the underlying provider.
"""


class LLMError(Exception):
    """Base for all giskard-llm errors."""

    def __init__(self, status_code: int, message: str, provider: str) -> None:
        self.status_code = status_code
        self.message = message
        self.provider = provider
        super().__init__(f"[{provider}] {status_code}: {message}")


class AuthenticationError(LLMError):
    """API key invalid or missing (401/403)."""


class RateLimitError(LLMError):
    """Rate limit exceeded (429)."""


class ServerError(LLMError):
    """Provider-side error (500/502/503/529)."""


class LLMTimeoutError(LLMError):
    """Request timed out (408)."""


class BadRequestError(LLMError):
    """Malformed request (400)."""


class UnsupportedOperationError(LLMError):
    """The provider does not support the requested operation."""

    def __init__(self, provider: str, operation: str) -> None:
        super().__init__(
            0, f"Provider '{provider}' does not support {operation}.", provider
        )
        self.operation = operation


class UnsupportedContentError(LLMError):
    """The provider returned content that giskard-llm cannot represent.

    Raised instead of silently dropping data. ``status_code`` is ``0`` so the
    retry middleware does not retry: the same request yields the same content.

    Parameters
    ----------
    provider : str
        Provider name (e.g. ``"anthropic"``).
    content_type : str
        Provider-native type of the unsupported item (e.g. ``"web_search_call"``).
    """

    def __init__(self, provider: str, content_type: str) -> None:
        super().__init__(
            0,
            f"Provider '{provider}' returned unsupported content type "
            f"'{content_type}'. Configure the provider with "
            "ignore_unsupported_content=True to drop it with a warning instead.",
            provider,
        )
        self.content_type = content_type


class ProviderNotAvailableError(LLMError):
    """The provider SDK is not installed."""

    def __init__(self, provider: str, package: str, extra: str | None = None) -> None:
        pip_extra = extra or provider
        super().__init__(
            0,
            f"Provider '{provider}' requires the '{package}' package. "
            f"Install it with: pip install giskard-llm[{pip_extra}]",
            provider,
        )
