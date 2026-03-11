"""Unit tests for generator serialization and deserialization."""

import uuid
from typing import override

from giskard.agents.chat import Message
from giskard.agents.generators import (
    BaseGenerator,
    FinishReason,
    GenerationParams,
    Generator,
)
from giskard.agents.generators.base import Response
from giskard.agents.generators.litellm_generator import (
    LiteLLMGenerator,
)
from giskard.agents.generators.middleware import RetryPolicy
from giskard.agents.templates import MessageTemplate
from giskard.agents.tools import Tool
from giskard.agents.workflow import ChatWorkflow, ErrorPolicy
from giskard.core import MinIntervalRateLimiter
from pydantic import Field


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
    assert isinstance(deserialized, LiteLLMGenerator)
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
            messages: list[Message],
            params: GenerationParams,
        ) -> tuple[Message, FinishReason]:
            return Message(role="assistant", content=self.content), "stop"

    original = CustomGenerator(content="Test response")
    serialized = original.model_dump_json()
    deserialized = BaseGenerator.model_validate_json(serialized)

    assert isinstance(deserialized, CustomGenerator)
    assert deserialized.content == "Test response"
    assert deserialized.kind == f"custom_test_{generator_id}"

    response = await deserialized.complete(
        messages=[Message(role="user", content="Test message")]
    )
    assert isinstance(response, Response)
    assert response.message.role == "assistant"
    assert response.message.content == "Test response"
    assert response.finish_reason == "stop"


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
    assert isinstance(deserialized.generator, LiteLLMGenerator)
    assert deserialized.generator.model == "test-model"

    assert deserialized.generator.retry_policy is not None
    assert deserialized.generator.retry_policy.max_attempts == 3
    assert deserialized.generator.retry_policy.base_delay == 1.0

    assert deserialized.generator.rate_limiter == rate_limiter

    assert len(deserialized.messages) == 2
    assert isinstance(deserialized.messages[0], MessageTemplate)
    assert deserialized.messages[0].role == "user"
    assert deserialized.messages[0].content_template == "Hello, how are you?"
    assert isinstance(deserialized.messages[1], MessageTemplate)
    assert deserialized.messages[1].role == "assistant"
    assert deserialized.messages[1].content_template == "I'm doing well!"

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
            messages: list[Message],
            params: GenerationParams,
        ) -> tuple[Message, FinishReason]:
            return Message(role="assistant", content=self.content), "stop"

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
    assert isinstance(deserialized.messages[0], MessageTemplate)
    assert deserialized.messages[0].content_template == "Test message"

    assert deserialized.inputs["test_input"] == "test_value"

    chat = await deserialized.run()
    assert chat.last.role == "assistant"
    assert chat.last.content == "Workflow test response"
