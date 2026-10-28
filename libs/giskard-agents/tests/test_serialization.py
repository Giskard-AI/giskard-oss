"""Unit tests for generator serialization and deserialization."""

import uuid
from collections.abc import Sequence
from typing import Any, override

import pytest
from giskard.agents.embeddings import BaseEmbeddingModel, LitellmEmbeddingModel
from giskard.agents.generators import (
    BaseGenerator,
    GenerationParams,
    Generator,
)
from giskard.agents.generators.giskard_llm_generator import (
    GiskardLLMGenerator,
)
from giskard.agents.generators.middleware import (
    CompletionMiddleware,
    RateLimiterMiddleware,
    RetryMiddleware,
    RetryPolicy,
)
from giskard.agents.tools import Tool
from giskard.agents.workflow import ChatWorkflow, ErrorPolicy
from giskard.core import MinIntervalRateLimiter
from giskard.core.discriminated import (
    _REGISTRY,
    Discriminated,
)
from giskard.llm.types import (
    AssistantMessage,
    ChatMessage,
    Choice,
    CompletionResponse,
)
from pydantic import Field, ValidationError


def test_generator_serialization():
    """Test basic generator serialization and deserialization."""
    rate_limiter = MinIntervalRateLimiter.from_rpm(100, max_concurrent=10)
    original = Generator(
        model="test-model",
        params=GenerationParams(
            temperature=0.5,
            max_tokens=100,
            response_format=None,
            tools=[Tool(name="test-tool", description="Test tool", fn=lambda: "test")],
        ),
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1.0),
        rate_limiter=rate_limiter,
    )
    serialized = original.model_dump_json(exclude={"params": {"tools"}})
    deserialized = BaseGenerator.model_validate_json(serialized)

    assert isinstance(deserialized, Generator)
    assert isinstance(deserialized, GiskardLLMGenerator)
    assert deserialized.model == "test-model"

    assert deserialized.retry_policy is not None
    assert deserialized.retry_policy.max_attempts == 3
    assert deserialized.retry_policy.base_delay == 1.0

    assert deserialized.rate_limiter == rate_limiter

    assert deserialized.params is not None
    assert deserialized.params.temperature == 0.5
    assert deserialized.params.max_tokens == 100
    assert deserialized.params.response_format is None


async def test_generator_serialization_custom_generator():
    """Test basic generator serialization and deserialization."""

    generator_id = str(uuid.uuid4())

    @BaseGenerator.register(f"custom_test_{generator_id}")
    class CustomGenerator(BaseGenerator):
        content: str = Field(description="The content of the response")

        @override
        async def _call_model(
            self,
            messages: Sequence[ChatMessage],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> CompletionResponse:
            return CompletionResponse(
                choices=[
                    Choice(
                        message=AssistantMessage(content=self.content),
                        finish_reason="stop",
                        index=0,
                    )
                ]
            )

    original = CustomGenerator(content="Test response")
    serialized = original.model_dump_json()
    deserialized = BaseGenerator.model_validate_json(serialized)

    assert isinstance(deserialized, CustomGenerator)
    assert deserialized.content == "Test response"
    assert deserialized.kind == f"custom_test_{generator_id}"

    response = await deserialized.complete(
        messages=[{"role": "user", "content": "Test message"}]
    )
    assert isinstance(response, CompletionResponse)
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Test response"
    assert response.choices[0].finish_reason == "stop"


def test_chat_workflow_serialization():
    """Test basic chat workflow serialization and deserialization."""
    rate_limiter = MinIntervalRateLimiter.from_rpm(100, max_concurrent=10)
    generator = Generator(
        model="test-model",
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1.0),
        rate_limiter=rate_limiter,
    )

    tool = Tool(name="test-tool", description="Test tool", fn=lambda: "test")

    original = (
        ChatWorkflow(generator=generator)
        .chat("Hello, how are you?", role="user")
        .chat("I'm doing well!", role="assistant")
        .with_tools(tool)
        .with_inputs(name="TestUser", value=42)
        .on_error(ErrorPolicy.RETURN)
    )

    serialized = original.model_dump_json(exclude={"tools"})
    deserialized = ChatWorkflow.model_validate_json(serialized)

    assert isinstance(deserialized.generator, Generator)
    assert isinstance(deserialized.generator, GiskardLLMGenerator)
    assert deserialized.generator.model == "test-model"

    assert deserialized.generator.retry_policy is not None
    assert deserialized.generator.retry_policy.max_attempts == 3
    assert deserialized.generator.retry_policy.base_delay == 1.0

    assert deserialized.generator.rate_limiter == rate_limiter

    assert len(deserialized.messages) == 2
    assert isinstance(deserialized.messages[0], ChatMessage)
    assert deserialized.messages[0].role == "user"
    assert deserialized.messages[0].content == "Hello, how are you?"
    assert isinstance(deserialized.messages[1], ChatMessage)
    assert deserialized.messages[1].role == "assistant"
    assert deserialized.messages[1].content == "I'm doing well!"

    assert deserialized.inputs["name"] == "TestUser"
    assert deserialized.inputs["value"] == 42

    assert deserialized.error_policy == ErrorPolicy.RETURN


async def test_chat_workflow_serialization_custom_generator():
    """Test chat workflow serialization with custom generator."""

    generator_id = str(uuid.uuid4())

    @BaseGenerator.register(f"custom_workflow_{generator_id}")
    class CustomGenerator(BaseGenerator):
        content: str = Field(description="The content of the response")

        @override
        async def _call_model(
            self,
            messages: Sequence[ChatMessage],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> CompletionResponse:
            return CompletionResponse(
                choices=[
                    Choice(
                        message=AssistantMessage(content=self.content),
                        finish_reason="stop",
                        index=0,
                    )
                ]
            )

    generator = CustomGenerator(content="Workflow test response")
    tool = Tool(
        name="workflow-tool",
        description="Workflow test tool",
        fn=lambda x: f"processed: {x}",
    )

    original = (
        ChatWorkflow(generator=generator)
        .chat("Test message", role="user")
        .with_tools(tool)
        .with_inputs(test_input="test_value")
    )

    serialized = original.model_dump_json(exclude={"tools"})
    deserialized = ChatWorkflow.model_validate_json(serialized)

    assert isinstance(deserialized.generator, CustomGenerator)
    assert deserialized.generator.content == "Workflow test response"
    assert deserialized.generator.kind == f"custom_workflow_{generator_id}"

    assert len(deserialized.messages) == 1
    assert isinstance(deserialized.messages[0], ChatMessage)
    assert deserialized.messages[0].content == "Test message"

    assert deserialized.inputs["test_input"] == "test_value"

    chat = await deserialized.run()
    assert chat.last.role == "assistant"
    assert chat.last.content == "Workflow test response"


# -- Unknown-key rejection (``extra="forbid"``) ------------------------------


def test_generator_rejects_unknown_key():
    """A typo'd key on a generator must fail loudly, not silently default."""
    payload = {
        "kind": "giskard_llm",
        "model": "gpt-4",
        # Typo: the real field is ``retry_policy``. Silently dropping this key
        # would leave retry_policy at its default instead of max_attempts=9.
        "retry_polcy": {"max_attempts": 9},
    }

    with pytest.raises(ValidationError, match="retry_polcy"):
        BaseGenerator.model_validate(payload)

    with pytest.raises(ValidationError, match="retry_polcy"):
        GiskardLLMGenerator.model_validate(payload)


def test_middleware_rejects_unknown_key():
    """A typo'd key on a middleware must fail loudly."""
    payload = {"kind": "retry", "retry_polcy": {"max_attempts": 9}}

    with pytest.raises(ValidationError, match="retry_polcy"):
        CompletionMiddleware.model_validate(payload)

    with pytest.raises(ValidationError, match="retry_polcy"):
        RetryMiddleware.model_validate(payload)


def test_middleware_rejects_unknown_key_nested_in_generator():
    """A typo inside ``BaseGenerator.middlewares`` must fail loudly."""
    payload = {
        "kind": "giskard_llm",
        "model": "gpt-4",
        "middlewares": [{"kind": "retry", "retry_polcy": {"max_attempts": 9}}],
    }

    with pytest.raises(ValidationError, match="retry_polcy"):
        BaseGenerator.model_validate(payload)


def test_embedding_model_rejects_unknown_key():
    """A typo'd ``model_name`` must not silently embed with the default model."""
    payload = {"kind": "litellm", "model_name": "text-embedding-3-small"}

    with pytest.raises(ValidationError, match="model_name"):
        BaseEmbeddingModel.model_validate(payload)

    with pytest.raises(ValidationError, match="model_name"):
        LitellmEmbeddingModel.model_validate(payload)


# -- ``kind`` discriminator round-trips --------------------------------------


def _registered_subclasses(base: type[Discriminated]) -> dict[str, type[Discriminated]]:
    return dict(_REGISTRY._subclasses[base])  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("base", "instances"),
    [
        (BaseGenerator, {"giskard_llm": GiskardLLMGenerator(model="m")}),
        (
            CompletionMiddleware,
            {
                "retry": RetryMiddleware(),
                "rate_limiter": RateLimiterMiddleware(
                    rate_limiter=MinIntervalRateLimiter.from_rpm(100)
                ),
            },
        ),
        (BaseEmbeddingModel, {"litellm": LitellmEmbeddingModel()}),
    ],
    ids=["generator", "middleware", "embedding"],
)
def test_kind_round_trips_under_forbid(
    base: type[Discriminated], instances: dict[str, Discriminated]
):
    """``model_dump()`` emits ``kind``; ``model_validate()`` must accept it back.

    Covers the polymorphic path (``Base.model_validate``) and the direct path
    (``Subclass.model_validate``), which bypasses the registry entirely.
    """
    for kind, instance in instances.items():
        dumped = instance.model_dump()
        assert dumped["kind"] == kind

        polymorphic = base.model_validate(dumped)
        assert type(polymorphic) is type(instance)
        assert polymorphic.kind == kind

        direct = type(instance).model_validate(dumped)
        assert type(direct) is type(instance)
        assert direct.kind == kind

        # JSON round trip too: the middleware module docstring advertises it.
        from_json = base.model_validate_json(instance.model_dump_json())
        assert type(from_json) is type(instance)


@pytest.mark.parametrize(
    "base",
    [BaseGenerator, CompletionMiddleware, BaseEmbeddingModel],
    ids=["generator", "middleware", "embedding"],
)
def test_every_registered_subclass_forbids_extra(base: type[Discriminated]):
    """No registered subclass may weaken ``extra`` back to allow/ignore.

    Pydantic merges ``model_config`` along the MRO, so a subclass declaring its
    own config keeps ``extra="forbid"`` unless it sets ``extra`` itself. Test
    doubles defined in other libs' suites may deliberately opt out, so only the
    production tree is asserted.
    """
    for kind, subclass in _registered_subclasses(base).items():
        if not subclass.__module__.startswith("giskard."):
            continue
        assert subclass.model_config.get("extra") == "forbid", (
            f"{subclass.__name__} (kind={kind!r}) resolves to "
            f"extra={subclass.model_config.get('extra')!r}"
        )
