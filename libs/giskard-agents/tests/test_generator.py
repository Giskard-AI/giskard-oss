import json
import time
from typing import Any, override
from unittest.mock import MagicMock, patch

import pytest
from giskard.agents.chat import Chat, Message
from giskard.agents.generators.base import BaseGenerator, GenerationParams, Response
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
from giskard.agents.templates import MessageTemplate
from giskard.agents.tools import Function, Tool, ToolCall, tool
from giskard.agents.workflow import ChatWorkflow
from giskard.core import MinIntervalRateLimiter
from litellm import ModelResponse
from pydantic import Field


@pytest.fixture
def mock_response():
    return ModelResponse(
        choices=[
            dict(
                finish_reason="stop",
                message=dict(role="assistant", content="Mock response"),
            )
        ]
    )


async def test_litellm_generator_completion_with_mock(
    generator: LiteLLMGenerator, mock_response: ModelResponse
):
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        response = await generator.complete(
            messages=[Message(role="user", content="Test message")]
        )

        assert response.message.role == "assistant"
        assert response.message.content == "Mock response"
        assert response.finish_reason == "stop"


@pytest.mark.functional
async def test_generator_completion(generator: LiteLLMGenerator):
    response = await generator.complete(
        messages=[
            Message(
                role="system",
                content="You are a helpful assistant, greeting the user with 'Hello I am TestBot'.",
            ),
            Message(role="user", content="Hello, world!"),
        ]
    )

    assert isinstance(response, Response)
    assert response.message.role == "assistant"
    assert isinstance(response.message.content, str)
    assert "I am TestBot" in response.message.content
    assert response.finish_reason == "stop"


@pytest.mark.functional
async def test_generator_chat(generator: LiteLLMGenerator):
    test_message = "Hello, world!"
    pipeline = generator.chat(test_message)

    assert isinstance(pipeline, ChatWorkflow)
    assert len(pipeline.messages) == 1
    assert isinstance(pipeline.messages[0], MessageTemplate)
    assert pipeline.messages[0].role == "user"
    assert pipeline.messages[0].content_template == test_message

    chat = await pipeline.run()

    assert isinstance(chat, Chat)

    chats = await pipeline.run_many(3)

    assert len(chats) == 3
    assert isinstance(chats[0], Chat)
    assert isinstance(chats[1], Chat)
    assert isinstance(chats[2], Chat)


async def test_litellm_generator_gets_rate_limiter(mock_response: ModelResponse):
    rate_limiter = MinIntervalRateLimiter.from_rpm(60, max_concurrent=1)
    generator = LiteLLMGenerator(
        model="test-model",
        rate_limiter=rate_limiter,
    )
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        start_time = time.monotonic()
        for _ in range(3):
            _ = await generator.complete(
                messages=[Message(role="user", content="Test message")]
            )
        end_time = time.monotonic()

    # Distribution of request:
    # t = 0.0 -> request 1
    # t = 1.0 -> request 2
    # t = 2.0 -> request 3
    elapsed_time = end_time - start_time
    assert elapsed_time >= 2
    assert elapsed_time < 3


async def test_generator_without_rate_limiter(mock_response: ModelResponse):
    generator = LiteLLMGenerator(model="test-model")
    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ):
        start_time = time.monotonic()
        for _ in range(3):
            _ = await generator.complete(
                messages=[Message(role="user", content="Test message")]
            )
        end_time = time.monotonic()

    elapsed_time = end_time - start_time
    assert elapsed_time < 10e-3  # arbitrary small number, here 10ms


def test_generator_with_params():
    generator = LiteLLMGenerator(model="test-model")
    generator = generator.with_params(temperature=0.5)
    assert generator.params.temperature == 0.5

    new_generator = generator.with_params(temperature=0.7)
    assert new_generator.params.temperature == 0.7
    assert generator.params.temperature == 0.5

    int_generator = new_generator.with_params(response_format=int)
    assert int_generator.params.response_format is int
    assert new_generator.params.response_format is None
    assert generator.params.response_format is None


def test_generator_with_params_and_rate_limiter():
    """with_params uses a shallow copy: rate limiter is shared, not deep-copied."""
    rate_limiter = MinIntervalRateLimiter.from_rpm(100, max_concurrent=5)
    generator = LiteLLMGenerator(
        model="test-model",
        rate_limiter=rate_limiter,
    )

    assert generator.rate_limiter == rate_limiter

    with patch.object(
        type(rate_limiter), "__deepcopy__", new_callable=MagicMock
    ) as mock_deepcopy:
        generator_with_params = generator.with_params(temperature=0.5, max_tokens=100)
        mock_deepcopy.assert_not_called()

    assert isinstance(generator_with_params, LiteLLMGenerator)
    assert generator_with_params.params.temperature == 0.5
    assert generator_with_params.params.max_tokens == 100

    assert generator_with_params.rate_limiter is rate_limiter

    assert generator.params.temperature == 1.0  # default value
    assert generator.params.max_tokens is None


async def test_generator_with_params_overwrite(mock_response: ModelResponse):
    # ARRANGE: Create a generator with base parameters.
    generator = LiteLLMGenerator(model="test-model").with_params(
        temperature=0.5,  # This should be preserved.
        max_tokens=100,  # This should be overwritten.
        timeout=30,  # This should be overwritten.
    )

    with patch(
        "giskard.agents.generators.litellm_generator.acompletion",
        return_value=mock_response,
    ) as mock_acompletion:
        # ACT: Call complete() with overriding parameters.
        _ = await generator.complete(
            messages=[Message(role="user", content="Test message")],
            params=GenerationParams(max_tokens=200, timeout=60),
        )

        # ASSERT: Verify that parameters were merged correctly.
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert (
            call_kwargs["temperature"] == 0.5
        )  # Preserved from the generator's params.
        assert (
            call_kwargs["max_tokens"] == 200
        )  # Overwritten by the complete() call's params.
        assert (
            call_kwargs["timeout"] == 60
        )  # Overwritten by the complete() call's params.
        assert call_kwargs["model"] == "test-model"


# ---------------------------------------------------------------------------
# Generator as protocol adapter — translation method overrides
# ---------------------------------------------------------------------------


class SpyGenerator(BaseGenerator):
    """A generator that records _call_model invocations and simulates
    a tool-calling LLM for one round."""

    canned_response: str = Field(default="done")
    calls: list[dict[str, Any]] = Field(default_factory=list)
    call_count: int = Field(default=0)

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        self.call_count += 1
        self.calls.append({"messages": messages, "params": params})

        if self.call_count == 1 and params.tools:
            return Response(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_spy_1",
                            function=Function(
                                name=params.tools[0].name,
                                arguments=json.dumps({"city": "Paris"}),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )

        return Response(
            message=Message(role="assistant", content=self.canned_response),
            finish_reason="stop",
        )


async def test_call_model_receives_internal_types():
    """Verify _call_model receives Message and Tool objects, not dicts."""

    @tool
    def get_weather(city: str) -> str:
        """Get weather.

        Parameters
        ----------
        city : str
            City name.
        """
        return f"Sunny in {city}"

    gen = SpyGenerator(canned_response="All done")
    chat = await (
        ChatWorkflow(generator=gen)
        .chat("What's the weather?", role="user")
        .with_tools(get_weather)
        .run()
    )

    assert len(gen.calls) >= 1
    assert all(isinstance(m, Message) for m in gen.calls[0]["messages"])
    assert isinstance(gen.calls[0]["params"], GenerationParams)
    assert all(isinstance(t, Tool) for t in gen.calls[0]["params"].tools)

    assert chat.last.content == "All done"

    tool_msg = next(m for m in chat.messages if m.role == "tool")
    assert tool_msg.content == "Sunny in Paris"
    assert tool_msg.tool_call_id == "call_spy_1"


async def test_subclass_controls_message_serialization():
    """A subclass can transform messages however it likes inside _call_model."""

    class TaggingGenerator(BaseGenerator):
        @override
        async def _call_model(
            self,
            messages: list[Message],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> Response:
            last_content = messages[-1].content or ""
            tagged = f"[tagged] {last_content}"
            return Response(
                message=Message(role="assistant", content=tagged), finish_reason="stop"
            )

    gen = TaggingGenerator()
    chat = await ChatWorkflow(generator=gen).chat("hello", role="user").run()

    assert chat.last.content == "[tagged] hello"


async def test_subclass_controls_tool_serialization():
    """A subclass can reshape tool definitions inside _call_model."""

    @tool
    def my_tool(x: str) -> str:
        """A tool.

        Parameters
        ----------
        x : str
            Input.
        """
        return x

    class RenamedToolGenerator(BaseGenerator):
        @override
        async def _call_model(
            self,
            messages: list[Message],
            params: GenerationParams,
            metadata: dict[str, Any] | None = None,
        ) -> Response:
            content = f"custom_{params.tools[0].name}" if params.tools else "none"
            return Response(
                message=Message(role="assistant", content=content), finish_reason="stop"
            )

    gen = RenamedToolGenerator()
    chat = await (
        ChatWorkflow(generator=gen).chat("hi", role="user").with_tools(my_tool).run()
    )

    assert chat.last.content == "custom_my_tool"
