"""Route ``provider/model`` strings to the correct provider instance."""

import importlib
import os
from collections.abc import Sequence
from typing import Any, Literal

from .errors import ProviderNotAvailableError, UnsupportedOperationError
from .providers.base import CompletionProvider, EmbeddingProvider, ResponseProvider
from .types import (
    ChatMessage,
    ChatMessageParam,
    CompletionResponse,
    EmbeddingResponse,
    ResponseInputItem,
    ResponseInputItemParam,
    ResponseResult,
    ToolDef,
    ToolDefParam,
)

# Plain assignment (not `type` statement) so isinstance checks work at runtime.
Provider = CompletionProvider | EmbeddingProvider | ResponseProvider

_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("giskard.llm.providers.openai", "OpenAIProvider"),
    "google": ("giskard.llm.providers.google", "GoogleProvider"),
    "gemini": ("giskard.llm.providers.google", "GoogleProvider"),
    "anthropic": ("giskard.llm.providers.anthropic", "AnthropicProvider"),
    "azure": ("giskard.llm.providers.azure_openai", "AzureOpenAIProvider"),
    "azure_ai": ("giskard.llm.providers.azure_ai", "AzureAIProvider"),
}


def _resolve_value(value: Any) -> Any:
    """Resolve ``os.environ/VAR_NAME`` strings to env var values."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        var_name = value[len("os.environ/") :]
        return os.environ.get(var_name)
    return value


def _parse_model_string(model: str) -> tuple[str, str]:
    """Split ``"provider/model-name"`` into ``(provider, model_name)``.

    Bare model names (no ``/``) default to ``"openai"``.
    """
    model = model.strip()
    if not model:
        raise ValueError(
            "Invalid model string ''. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )
    if "/" not in model:
        return "openai", model
    parts = model.split("/", maxsplit=1)
    provider, model_name = parts[0].strip(), parts[1].strip()
    if not provider or not model_name:
        raise ValueError(
            f"Invalid model string '{model}'. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )
    return provider, model_name


def _create_provider(provider_type: str, **kwargs: Any) -> Provider:
    """Instantiate a provider by type name using the registry."""
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider_type}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_REGISTRY))}."
        )
    module_path, class_name = _PROVIDER_REGISTRY[provider_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def _probe_provider_sdk(provider_type: str) -> bool:
    """Return whether the provider module's SDK import helpers succeed."""
    module_path, _ = _PROVIDER_REGISTRY[provider_type]
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return False

    for name in sorted(dir(module)):
        if not name.startswith("_import_") or name.endswith(("_errors", "_types")):
            continue
        importer = getattr(module, name, None)
        if not callable(importer):
            continue
        try:
            importer()
            return True
        except (ImportError, ProviderNotAvailableError):
            return False

    try:
        openai_module = importlib.import_module("giskard.llm.providers.openai")
        openai_module._import_openai()
    except (ImportError, ProviderNotAvailableError, AttributeError):
        return False
    return True


class LLMClient:
    """Entry point for configuring and calling LLM providers.

    Stores config from ``configure()`` calls. Provider instances are
    created lazily on first use and cached on this client instance.
    """

    def __init__(self) -> None:
        self._configs: dict[str, dict[str, Any]] = {}
        self._providers: dict[str, Provider] = {}

    def configure(self, name: str, provider: str | None = None, **kwargs: Any) -> None:
        """Register a named provider configuration.

        Args:
            name: Alias for this provider (used as model string prefix).
            provider: Provider type from the registry. Defaults to *name*.
            **kwargs: Connection config (api_key, base_url, ...) and
                behavior config (merge_system, ...). Values may use
                ``os.environ/VAR_NAME`` syntax for deferred env var resolution.
        """
        name = name.strip()
        provider_type = (provider or name).strip()
        self._configs[name] = {"provider": provider_type, **kwargs}
        self._providers.pop(name, None)

    def configure_from_dict(self, config: dict[str, dict[str, Any]]) -> None:
        """Bulk-register providers from a dict (e.g. loaded from YAML)."""
        for name, kwargs in config.items():
            self.configure(name, **kwargs)

    def _resolve_provider_type(self, alias: str) -> str:
        """Return the registry provider type for *alias*."""
        if alias in self._configs:
            return self._configs[alias]["provider"]
        if alias in _PROVIDER_REGISTRY:
            return alias
        raise ValueError(
            f"Provider '{alias}' is not configured and not in the registry. "
            f"Call client.configure('{alias}', ...) first."
        )

    def _get_provider(self, name: str) -> Provider:
        if name in self._providers:
            return self._providers[name]

        provider_type = self._resolve_provider_type(name)
        if name in self._configs:
            cfg = dict(self._configs[name])
            cfg.pop("provider")
            resolved = {k: _resolve_value(v) for k, v in cfg.items()}
            provider = _create_provider(provider_type, **resolved)
        else:
            provider = _create_provider(provider_type)
        self._providers[name] = provider
        return provider

    def supports_native(
        self, model: str, operation: Literal["completion", "embedding"]
    ) -> bool:
        """Return whether this client can handle *model* for *operation* natively."""
        method_name = "complete" if operation == "completion" else "embed"
        try:
            alias, _ = _parse_model_string(model)
            provider_type = self._resolve_provider_type(alias)
        except ValueError:
            return False

        if provider_type not in _PROVIDER_REGISTRY:
            return False

        module_path, class_name = _PROVIDER_REGISTRY[provider_type]
        try:
            module = importlib.import_module(module_path)
            provider_cls = getattr(module, class_name)
        except ImportError:
            return False

        if not callable(getattr(provider_cls, method_name, None)):
            return False

        return _probe_provider_sdk(provider_type)

    def _resolve(self, model: str, protocol: type, operation: str) -> tuple[Any, str]:
        """Parse model string, look up the provider, and check capability."""
        alias, model_name = _parse_model_string(model)
        provider = self._get_provider(alias)
        if not isinstance(provider, protocol):
            raise UnsupportedOperationError(alias, operation)
        return provider, model_name

    async def acompletion(
        self,
        model: str,
        messages: Sequence[ChatMessageParam | ChatMessage],
        *,
        tools: Sequence[ToolDefParam | ToolDef] | None = None,
        **params: Any,
    ) -> CompletionResponse:
        """Parse model string and dispatch to the right provider."""
        provider, model_name = self._resolve(model, CompletionProvider, "completions")
        return await provider.complete(model_name, messages, tools=tools, **params)

    async def aembedding(
        self,
        model: str,
        input: list[str],
        **params: Any,
    ) -> EmbeddingResponse:
        """Parse model string and dispatch to the right provider."""
        provider, model_name = self._resolve(model, EmbeddingProvider, "embeddings")
        return await provider.embed(model_name, input, **params)

    async def aresponse(
        self,
        model: str,
        input: str | Sequence[ResponseInputItemParam | ResponseInputItem],
        *,
        instructions: str | None = None,
        previous_id: str | None = None,
        tools: Sequence[ToolDefParam | ToolDef] | None = None,
        **params: Any,
    ) -> ResponseResult:
        """Parse model string and dispatch to the right provider's respond()."""
        provider, model_name = self._resolve(
            model, ResponseProvider, "the Responses/Interactions API"
        )
        return await provider.respond(
            model_name,
            input,
            instructions=instructions,
            previous_id=previous_id,
            tools=tools,
            **params,
        )


_default_client = LLMClient()


def supports_native(
    model: str,
    operation: Literal["completion", "embedding"],
    *,
    client: LLMClient | None = None,
) -> bool:
    """Return whether giskard-llm can handle *model* for *operation* natively.

    Checks registry membership (or ``configure()`` aliases), SDK availability,
    and provider protocol support. Does not perform network I/O.
    """
    return (client or _default_client).supports_native(model, operation)


def configure(name: str, provider: str | None = None, **kwargs: Any) -> None:
    """Configure a provider on the default client."""
    _default_client.configure(name, provider, **kwargs)


def reset() -> None:
    """Clear all cached providers on the default client (useful in tests)."""
    _default_client._providers.clear()
    _default_client._configs.clear()


async def acompletion(
    model: str,
    messages: Sequence[ChatMessageParam | ChatMessage],
    *,
    tools: Sequence[ToolDefParam | ToolDef] | None = None,
    **params: Any,
) -> CompletionResponse:
    """Module-level convenience wrapper around the default client."""
    return await _default_client.acompletion(model, messages, tools=tools, **params)


async def aembedding(
    model: str,
    input: list[str],
    **params: Any,
) -> EmbeddingResponse:
    """Module-level convenience wrapper around the default client."""
    return await _default_client.aembedding(model, input, **params)


async def aresponse(
    model: str,
    input: str | Sequence[ResponseInputItemParam | ResponseInputItem],
    *,
    instructions: str | None = None,
    previous_id: str | None = None,
    tools: Sequence[ToolDefParam | ToolDef] | None = None,
    **params: Any,
) -> ResponseResult:
    """Module-level convenience wrapper around the default client."""
    return await _default_client.aresponse(
        model,
        input,
        instructions=instructions,
        previous_id=previous_id,
        tools=tools,
        **params,
    )
