"""System One provider routing, transport, and serialization contracts."""

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, override

import httpx
import pytest
from giskard.agents import BaseSOM, SOMResponse, resolve_som
from giskard.agents.som import TypeSafeSOM
from giskard.llm.types import ChatMessage, SystemMessage, UserMessage
from pydantic import ValidationError

RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {"decision": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 300, "output_tokens": 20},
}
MESSAGES: list[ChatMessage] = [
    SystemMessage(content="A bank card replacement costs 12 EUR."),
    UserMessage(content="I paid 12 EUR to replace my lost bank card."),
]
QUESTION = "Does the customer's payment match the bank's policy?"


@pytest.fixture
def mock_api(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[httpx.Request]]:
    """Use real HTTP serialization with an in-memory transport."""
    client_class = httpx.AsyncClient
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-secret")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)

    def configure(payload: Any = None, *, status: int = 200) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                status,
                # Include NaN in malformed responses to exercise validation.
                content=json.dumps(RESPONSE if payload is None else payload),
            )

        def create_client(**kwargs: Any) -> httpx.AsyncClient:
            return client_class(transport=httpx.MockTransport(handle), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", create_client)
        return requests

    return configure


async def test_native_prediction_preserves_question_context_and_usage(mock_api):
    requests = mock_api()

    prediction = await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)

    assert prediction.probability == 0.9
    assert prediction.model == "jev-1.13.0"
    assert prediction.usage is not None
    assert prediction.usage.model_dump() == {
        "input_tokens": 300,
        "output_tokens": 20,
        "total_tokens": 320,
    }
    body = json.loads(requests[0].content)
    assert body == {
        "model": "jev-latest",
        "state": [
            {"role": "system", "content": MESSAGES[0].text},
            {"role": "user", "content": MESSAGES[1].text},
        ],
        "questions": {"decision": {"type": "noul", "instructions": QUESTION}},
    }
    assert requests[0].headers["Authorization"] == "Bearer test-typesafe-secret"
    assert requests[0].extensions["timeout"]["read"] == 30


@pytest.mark.parametrize("probability", [0, 0.5, 1])
async def test_prediction_preserves_probability_without_a_threshold(
    mock_api, probability
):
    payload = deepcopy(RESPONSE)
    payload["answers"]["decision"]["noul"] = probability
    mock_api(payload)

    prediction = await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)

    assert prediction.probability == probability


@pytest.mark.parametrize(
    "probability", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9", None]
)
async def test_invalid_probability_is_rejected(mock_api, probability):
    payload = deepcopy(RESPONSE)
    payload["answers"]["decision"]["noul"] = probability
    mock_api(payload)

    with pytest.raises(ValidationError):
        await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)

    with pytest.raises(ValidationError):
        SOMResponse(probability=probability, model="custom-model")


@pytest.mark.parametrize("missing_field", ["model", "answers", "usage"])
async def test_incomplete_response_is_rejected(mock_api, missing_field):
    payload = deepcopy(RESPONSE)
    del payload[missing_field]
    mock_api(payload)

    with pytest.raises(ValidationError):
        await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"other_question": {"type": "noul", "noul": 0.9}},
        {"decision": {"type": "score", "noul": 0.9}},
    ],
)
async def test_missing_or_wrong_answer_type_is_rejected(mock_api, answer):
    mock_api({**RESPONSE, "answers": answer})

    with pytest.raises(ValidationError):
        await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)


@pytest.mark.parametrize("tokens", [-1, 1.5, True, "20"])
async def test_invalid_usage_is_rejected(mock_api, tokens):
    payload = deepcopy(RESPONSE)
    payload["usage"]["input_tokens"] = tokens
    mock_api(payload)

    with pytest.raises(ValidationError):
        await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_failure_propagates(mock_api, status):
    mock_api({"error": "provider failure"}, status=status)

    with pytest.raises(httpx.HTTPStatusError) as error:
        await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)

    assert error.value.response.status_code == status


async def test_missing_credentials_fail_only_when_called(mock_api, monkeypatch):
    requests = mock_api()
    monkeypatch.delenv("TYPESAFE_API_KEY")
    provider = TypeSafeSOM(model="jev")
    restored = BaseSOM.model_validate_json(provider.model_dump_json())
    assert isinstance(restored, TypeSafeSOM)

    with pytest.raises(ValueError, match="Set TYPESAFE_API_KEY"):
        await restored.predict(MESSAGES, QUESTION)

    assert requests == []


@pytest.mark.parametrize(
    "base, endpoint",
    [
        ("https://api.typesafe.ai", "https://api.typesafe.ai/v1/systemone"),
        ("https://api.typesafe.ai/v1/", "https://api.typesafe.ai/v1/systemone"),
        (
            "https://api.typesafe.ai/v1/systemone/",
            "https://api.typesafe.ai/v1/systemone",
        ),
        (
            "http://localhost:4000/typesafe",
            "http://localhost:4000/typesafe/v1/systemone",
        ),
        (
            "http://localhost:4000/typesafe/v1/systemone",
            "http://localhost:4000/typesafe/v1/systemone",
        ),
    ],
)
async def test_direct_and_gateway_endpoint_shapes(mock_api, base, endpoint):
    requests = mock_api()

    await TypeSafeSOM(model="jev", base_url=base).predict(MESSAGES, QUESTION)

    assert str(requests[0].url) == endpoint


async def test_environment_url_and_gateway_credentials(mock_api, monkeypatch):
    requests = mock_api()
    monkeypatch.setenv("TYPESAFE_API_BASE", "https://api.typesafe.ai")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:4000/typesafe/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "test-gateway-secret")
    provider = TypeSafeSOM(model="jev", api_key_env="LITELLM_API_KEY")

    await provider.predict(MESSAGES, QUESTION, timeout=7)

    assert str(requests[0].url) == "http://localhost:4000/typesafe/v1/systemone"
    assert requests[0].headers["Authorization"] == "Bearer test-gateway-secret"
    assert requests[0].extensions["timeout"]["read"] == 7
    serialized = provider.model_dump_json()
    assert "test-gateway-secret" not in serialized
    restored = BaseSOM.model_validate_json(serialized)
    assert isinstance(restored, TypeSafeSOM)
    assert restored.api_key_env == "LITELLM_API_KEY"


async def test_api_base_fallback_and_explicit_url_precedence(mock_api, monkeypatch):
    requests = mock_api()
    monkeypatch.setenv("TYPESAFE_API_BASE", "https://fallback.example/v1")
    await TypeSafeSOM(model="jev").predict(MESSAGES, QUESTION)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://environment.example/v1")
    await TypeSafeSOM(model="jev", base_url="https://explicit.example/v1").predict(
        MESSAGES, QUESTION
    )

    assert str(requests[0].url) == "https://fallback.example/v1/systemone"
    assert str(requests[1].url) == "https://explicit.example/v1/systemone"


@pytest.mark.parametrize("model", ["jev-latest", "jev-1.13.0", "future-model"])
def test_native_model_names_are_preserved(model):
    provider = resolve_som(f"typesafe/{model}")

    assert isinstance(provider, TypeSafeSOM)
    assert provider.model == model


def test_provider_alias_resolves_to_native_model():
    provider = resolve_som("typesafe/jev")

    assert isinstance(provider, TypeSafeSOM)
    assert provider.model == "jev-latest"


@pytest.mark.parametrize("model", ["typesafe", "typesafe/", "typesafe/ "])
def test_missing_supported_provider_model_is_rejected(model):
    with pytest.raises(ValueError, match="provider/model"):
        resolve_som(model)


@pytest.mark.parametrize("model", ["azure_ai/gpt-5.6-luna", "gpt-4o", "other/model"])
def test_unknown_provider_allows_other_model_routing(model):
    assert resolve_som(model) is None


def test_model_configuration_requires_model_and_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TypeSafeSOM.model_validate({})

    with pytest.raises(ValidationError):
        TypeSafeSOM.model_validate({"model": ""})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TypeSafeSOM.model_validate({"model": "jev", "model_name": "other"})


async def test_another_provider_uses_the_same_interface(monkeypatch):
    from giskard.agents.som import _PROVIDERS

    @BaseSOM.register("test_som_provider")
    class OtherSOM(BaseSOM):
        @override
        async def predict(
            self,
            messages: Sequence[ChatMessage],
            question: str,
            *,
            timeout: float | int | None = None,
        ) -> SOMResponse:
            return SOMResponse(probability=0.75, model=self.model)

    monkeypatch.setitem(_PROVIDERS, "other", OtherSOM)
    provider = resolve_som("other/a-model")
    assert isinstance(provider, OtherSOM)
    restored = BaseSOM.model_validate_json(provider.model_dump_json())

    response = await restored.predict(MESSAGES, QUESTION)

    assert isinstance(restored, OtherSOM)
    assert response.probability == 0.75
    assert response.model == "a-model"
    assert response.usage is None


def test_agents_som_import_does_not_require_checks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from giskard.agents import resolve_som; "
            "assert resolve_som('typesafe/jev') is not None; "
            "assert not any(name.startswith('giskard.checks') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GISKARD_TELEMETRY_DISABLED": "1"},
    )

    assert result.returncode == 0, result.stderr
