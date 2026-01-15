"""Unit tests for generator serialization and deserialization."""

import uuid

from giskard.agents.chat import Message
from giskard.agents.generators import BaseGenerator, GenerationParams, Generator
from giskard.agents.generators.base import Response
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
from giskard.agents.generators.retry import RetryPolicy
from giskard.agents.rate_limiter import RateLimiter
from giskard.agents.templates import MessageTemplate
from giskard.agents.tools import Tool
from giskard.agents.workflow import ChatWorkflow, ErrorPolicy
from pydantic import Field


def test_generator_serialization():
    """Test basic generator serialization and deserialization."""
    original = Generator(
        model="test-model",
        retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
        rate_limiter=RateLimiter.from_rpm(rpm=100, max_concurrent=10),
        params=GenerationParams(
            temperature=0.5,
            max_tokens=100,
            response_format=None,
            tools=[Tool(name="test-tool", description="Test tool", fn=lambda: "test")],
        ),
    )
    serialized = original.model_dump_json(exclude={"params": {"tools"}})
    deserialized = BaseGenerator.model_validate_json(serialized)

    assert isinstance(deserialized, Generator)
    assert isinstance(deserialized, LiteLLMGenerator)
    assert deserialized.model == "test-model"

    assert deserialized.retry_policy is not None
    assert deserialized.retry_policy.max_retries == 3
    assert deserialized.retry_policy.base_delay == 1.0

    assert deserialized.rate_limiter is not None
    assert deserialized.rate_limiter.strategy.min_interval == 0.6
    assert deserialized.rate_limiter.strategy.max_concurrent == 10

    assert deserialized.params is not None
    assert deserialized.params.temperature == 0.5
    assert deserialized.params.max_tokens == 100
    assert deserialized.params.response_format is None


async def test_generator_serialization_custom_generator():
    """Test basic generator serialization and deserialization."""

    # Ensure the generator is registered with a unique name
    generator_id = str(uuid.uuid4())

    @BaseGenerator.register(f"custom_test_{generator_id}")
    class CustomGenerator(BaseGenerator):
        content: str = Field(description="The content of the response")

        async def _complete(
            self, messages: list[Message], params: GenerationParams | None = None
        ) -> Response:
            return Response(
                message=Message(role="assistant", content=self.content),
                finish_reason="stop",
            )

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
    generator = Generator(
        model="test-model",
        retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
        rate_limiter=RateLimiter.from_rpm(rpm=100, max_concurrent=10),
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

    # Verify generator is restored
    assert isinstance(deserialized.generator, Generator)
    assert isinstance(deserialized.generator, LiteLLMGenerator)
    assert deserialized.generator.model == "test-model"

    # Verify messages are restored
    assert len(deserialized.messages) == 2
    assert isinstance(deserialized.messages[0], MessageTemplate)
    assert deserialized.messages[0].role == "user"
    assert deserialized.messages[0].content_template == "Hello, how are you?"
    assert isinstance(deserialized.messages[1], MessageTemplate)
    assert deserialized.messages[1].role == "assistant"
    assert deserialized.messages[1].content_template == "I'm doing well!"

    # Verify inputs are restored
    assert deserialized.inputs["name"] == "TestUser"
    assert deserialized.inputs["value"] == 42

    # Verify error policy is restored
    assert deserialized.error_policy == ErrorPolicy.RETURN


async def test_chat_workflow_serialization_custom_generator():
    """Test chat workflow serialization with custom generator."""

    # Ensure the generator is registered with a unique name
    generator_id = str(uuid.uuid4())

    @BaseGenerator.register(f"custom_workflow_{generator_id}")
    class CustomGenerator(BaseGenerator):
        content: str = Field(description="The content of the response")

        async def _complete(
            self, messages: list[Message], params: GenerationParams | None = None
        ) -> Response:
            return Response(
                message=Message(role="assistant", content=self.content),
                finish_reason="stop",
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

    # Verify generator is restored correctly
    assert isinstance(deserialized.generator, CustomGenerator)
    assert deserialized.generator.content == "Workflow test response"
    assert deserialized.generator.kind == f"custom_workflow_{generator_id}"

    # Verify messages are restored
    assert len(deserialized.messages) == 1
    assert isinstance(deserialized.messages[0], MessageTemplate)
    assert deserialized.messages[0].content_template == "Test message"

    # Verify inputs are restored
    assert deserialized.inputs["test_input"] == "test_value"

    # Test that the workflow can still run after deserialization
    chat = await deserialized.run()
    assert chat.last.role == "assistant"
    assert chat.last.content == "Workflow test response"
