"""Tests for model string parsing, registry, and LLMClient routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from giskard.llm.routing import (
    LLMClient,
    _create_provider,
    _parse_model_string,
    _resolve_value,
)

# -- _parse_model_string -------------------------------------------------------


@pytest.mark.parametrize(
    "model_str, expected",
    [
        ("openai/gpt-4o", ("openai", "gpt-4o")),
        ("google/gemini-2.0-flash", ("google", "gemini-2.0-flash")),
        ("gemini/gemini-2.0-flash", ("gemini", "gemini-2.0-flash")),
        ("anthropic/claude-opus-4-6", ("anthropic", "claude-opus-4-6")),
        ("azure/gpt-4o", ("azure", "gpt-4o")),
        ("azure_ai/gpt-4o", ("azure_ai", "gpt-4o")),
    ],
    ids=["openai", "google", "gemini-alias", "anthropic", "azure", "azure_ai"],
)
def test_parse_model_string_valid(model_str: str, expected: tuple[str, str]):
    assert _parse_model_string(model_str) == expected


@pytest.mark.parametrize(
    "model_str, expected",
    [
        ("gpt-4o", ("openai", "gpt-4o")),
        ("gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("claude-opus-4-6", ("openai", "claude-opus-4-6")),
    ],
    ids=["bare-gpt4o", "bare-gpt4o-mini", "bare-claude"],
)
def test_parse_model_string_no_prefix_defaults_to_openai(
    model_str: str, expected: tuple[str, str]
):
    assert _parse_model_string(model_str) == expected


@pytest.mark.parametrize(
    "model_str",
    ["", "  ", "openai/", "/gpt-4o", " / "],
    ids=["empty", "whitespace-only", "no-model", "no-provider", "slashes-whitespace"],
)
def test_parse_model_string_invalid(model_str: str):
    with pytest.raises(ValueError, match="Invalid model string"):
        _parse_model_string(model_str)


@pytest.mark.parametrize(
    "model_str, expected",
    [
        ("  openai/gpt-4o  ", ("openai", "gpt-4o")),
        ("openai / gpt-4o", ("openai", "gpt-4o")),
        ("  gpt-4o  ", ("openai", "gpt-4o")),
    ],
    ids=["outer-whitespace", "inner-whitespace", "bare-whitespace"],
)
def test_parse_model_string_strips_whitespace(
    model_str: str, expected: tuple[str, str]
):
    assert _parse_model_string(model_str) == expected


# -- _resolve_value ------------------------------------------------------------


def test_resolve_value_plain_string():
    assert _resolve_value("sk-abc") == "sk-abc"


def test_resolve_value_non_string():
    assert _resolve_value(42) == 42
    assert _resolve_value(True) is True


def test_resolve_value_env_var(monkeypatch):
    monkeypatch.setenv("TEST_KEY_123", "resolved-value")
    assert _resolve_value("os.environ/TEST_KEY_123") == "resolved-value"


def test_resolve_value_env_var_missing():
    assert _resolve_value("os.environ/NONEXISTENT_VAR_XYZ") is None


# -- _create_provider ----------------------------------------------------------


def test_create_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider 'foobar'"):
        _create_provider("foobar")


# -- LLMClient ----------------------------------------------------------------


def test_client_configure_and_lookup():
    client = LLMClient()
    client.configure(
        "my-openai", provider="openai", api_key="sk-test"
    )  # pragma: allowlist secret
    assert "my-openai" in client._configs
    assert client._configs["my-openai"]["provider"] == "openai"


def test_client_configure_defaults_provider_to_name():
    client = LLMClient()
    client.configure("openai", api_key="sk-test")  # pragma: allowlist secret
    assert client._configs["openai"]["provider"] == "openai"


def test_client_configure_from_dict():
    client = LLMClient()
    client.configure_from_dict(
        {
            "my-openai": {"provider": "openai", "api_key": "sk-1"},
            "my-google": {"provider": "google", "api_key": "gk-1"},
        }
    )
    assert "my-openai" in client._configs
    assert "my-google" in client._configs


def test_client_configure_env_resolution(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "resolved-key")
    client = LLMClient()
    client.configure(
        "test-openai", provider="openai", api_key="os.environ/MY_API_KEY"
    )  # pragma: allowlist secret

    with patch("giskard.llm.routing._create_provider") as mock_create:
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        provider = client._get_provider("test-openai")
        mock_create.assert_called_once_with(
            "openai", api_key="resolved-key"
        )  # pragma: allowlist secret
        assert provider is mock_provider


def test_client_named_aliases_route_correctly():
    client = LLMClient()
    client.configure(
        "azure-a", provider="openai", api_key="key-a"
    )  # pragma: allowlist secret
    client.configure(
        "azure-b", provider="openai", api_key="key-b"
    )  # pragma: allowlist secret

    with patch("giskard.llm.routing._create_provider") as mock_create:
        provider_a = MagicMock()
        provider_b = MagicMock()
        mock_create.side_effect = [provider_a, provider_b]

        assert client._get_provider("azure-a") is provider_a
        assert client._get_provider("azure-b") is provider_b
        assert mock_create.call_count == 2


def test_client_lazy_caching():
    client = LLMClient()
    client.configure(
        "test", provider="openai", api_key="sk-x"
    )  # pragma: allowlist secret

    with patch("giskard.llm.routing._create_provider") as mock_create:
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        p1 = client._get_provider("test")
        p2 = client._get_provider("test")
        assert p1 is p2
        mock_create.assert_called_once()


def test_client_reconfigure_clears_cache():
    client = LLMClient()
    client.configure(
        "test", provider="openai", api_key="sk-1"
    )  # pragma: allowlist secret

    with patch("giskard.llm.routing._create_provider") as mock_create:
        mock_create.return_value = MagicMock()
        client._get_provider("test")

        client.configure(
            "test", provider="openai", api_key="sk-2"
        )  # pragma: allowlist secret
        assert "test" not in client._providers


def test_client_unconfigured_registry_provider():
    """Providers in the registry can be used without explicit configure()."""
    client = LLMClient()
    with patch("giskard.llm.routing._create_provider") as mock_create:
        mock_create.return_value = MagicMock()
        client._get_provider("openai")
        mock_create.assert_called_once_with("openai")


def test_client_unknown_provider_raises():
    client = LLMClient()
    with pytest.raises(ValueError, match="not configured"):
        client._get_provider("nonexistent")


@patch("giskard.llm.routing._create_provider")
async def test_client_acompletion_routes(mock_create):
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=MagicMock(choices=[]))
    mock_create.return_value = mock_provider

    client = LLMClient()
    await client.acompletion("openai/gpt-4o", [{"role": "user", "content": "Hi"}])
    mock_provider.complete.assert_called_once_with(
        "gpt-4o", [{"role": "user", "content": "Hi"}], tools=None
    )


@patch("giskard.llm.routing._create_provider")
async def test_client_aembedding_routes(mock_create):
    mock_provider = MagicMock()
    mock_provider.embed = AsyncMock(return_value=MagicMock(data=[]))
    mock_create.return_value = mock_provider

    client = LLMClient()
    await client.aembedding("openai/text-embedding-3-small", ["hello"])
    mock_provider.embed.assert_called_once_with("text-embedding-3-small", ["hello"])


@patch("giskard.llm.routing._create_provider")
async def test_client_aresponse_dispatches(mock_create):
    mock_provider = MagicMock()
    mock_provider.respond = AsyncMock(return_value=MagicMock(id="resp_1"))
    mock_create.return_value = mock_provider

    client = LLMClient()
    await client.aresponse("openai/gpt-4o", "Hello")
    mock_provider.respond.assert_called_once_with(
        "gpt-4o", "Hello", instructions=None, previous_id=None, tools=None
    )


@patch("giskard.llm.routing._create_provider")
async def test_client_aresponse_unsupported_raises(mock_create):
    """Provider without respond() method raises UnsupportedOperationError."""
    from giskard.llm.errors import UnsupportedOperationError

    provider = MagicMock(spec=["complete", "embed"])
    mock_create.return_value = provider

    client = LLMClient()
    with pytest.raises(UnsupportedOperationError, match="does not support"):
        await client.aresponse("openai/gpt-4o", "Hello")
