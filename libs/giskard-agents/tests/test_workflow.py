import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast, override

import pytest
from giskard import agents
from giskard.agents.chat import Chat
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.agents.generators.giskard_llm_generator import GiskardLLMGenerator
from giskard.agents.templates.prompts_manager import PromptsManager
from giskard.llm.types import AssistantMessage, ChatMessage, Choice, CompletionResponse
from pydantic import BaseModel, Field


@pytest.mark.google
@pytest.mark.functional
async def test_single_run(generator):
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await (
        workflow.chat("Your name is TestBot.", role="system")
        .chat("What is your name? Answer in one word.", role="user")
        .run()
    )

    assert chat.last.text is not None
    assert "testbot" in chat.last.text.lower()


@pytest.mark.google
@pytest.mark.functional
async def test_run_many(generator):
    """Test that the workflow runs correctly."""

    workflow = agents.ChatWorkflow(generator=generator)

    chats = await workflow.chat("Hello!", role="user").run_many(n=3)

    assert len(chats) == 3


@pytest.mark.google
@pytest.mark.functional
async def test_run_batch(generator):
    """Test that the workflow runs correctly."""

    workflow = agents.ChatWorkflow(generator=generator)

    chats = await workflow.chat(
        "Hello {{ n }}!", role="user", as_template=True
    ).run_batch(inputs=[{"n": i} for i in range(3)])

    assert chats[0].context.inputs["n"] == 0
    assert chats[1].context.inputs["n"] == 1
    assert chats[2].context.inputs["n"] == 2

    assert chats[0].messages[0].content == "Hello 0!"
    assert chats[1].messages[0].content == "Hello 1!"
    assert chats[2].messages[0].content == "Hello 2!"

    assert len(chats) == 3


@pytest.mark.google
@pytest.mark.functional
async def test_stream_many(generator):
    workflow = agents.ChatWorkflow(generator=generator).chat("Hello!", role="user")

    chats = []
    async for chat in workflow.stream_many(3):
        assert isinstance(chat, Chat)
        chats.append(chat)

    assert len(chats) == 3


@pytest.mark.google
@pytest.mark.functional
async def test_stream_batch(generator):
    workflow = agents.ChatWorkflow(generator=generator).chat("Hello!", role="user")

    chats = []
    async for chat in workflow.stream_batch(
        inputs=[{"message": "Hello!"}, {"message": "Hello!!"}]
    ):
        assert isinstance(chat, Chat)
        chats.append(chat)

    assert "Hello!" in [c.context.inputs["message"] for c in chats]
    assert "Hello!!" in [c.context.inputs["message"] for c in chats]

    assert len(chats) == 2


@pytest.mark.google
@pytest.mark.functional
async def test_workflow_with_mixed_templates(generator: GiskardLLMGenerator):
    workflow = agents.ChatWorkflow(
        generator=generator,
        prompt_manager=PromptsManager(
            default_prompts_path=Path(__file__).parent / "data" / "prompts"
        ),
    )

    chat = (
        await workflow.template("multi_message.j2")
        .chat("{{ score }}!", role="assistant", as_template=True)
        .chat("Well done {{ name }}!", role="user", as_template=True)
        .with_inputs(
            name="TestBot",
            theory="Normandy is actually the center of the universe.",
            score=100,
        )
        .run()
    )

    assert len(chat.messages) == 5

    assert chat.messages[0].role == "system"
    assert chat.messages[0].text is not None
    assert (
        "You are an impartial evaluator of scientific theories."
        in chat.messages[0].text
    )

    assert chat.messages[1].role == "user"
    assert chat.messages[1].text is not None
    assert "Normandy is actually the center of the universe." in chat.messages[1].text

    assert chat.messages[2].role == "assistant"
    assert chat.messages[2].text is not None
    assert "100" in chat.messages[2].text

    assert chat.messages[3].role == "user"
    assert chat.messages[3].text is not None
    assert "Well done TestBot!" in chat.messages[3].text

    assert chat.messages[4].role == "assistant"


@pytest.mark.google
@pytest.mark.functional
async def test_output_format(generator):
    workflow = agents.ChatWorkflow(generator=generator)

    class SimpleOutput(BaseModel):
        mood: str
        greeting: str

    chat = (
        await workflow.chat("Hello! Answer in JSON.", role="user")
        .with_output(SimpleOutput)
        .run()
    )

    assert isinstance(chat.output, SimpleOutput)


def test_chat_plain_string_is_not_jinja_by_default():
    """Regression for ENG-1488: user-controlled strings must not be Jinja source by default."""
    workflow = agents.ChatWorkflow(
        generator=agents.Generator(model="openai/gpt-4o-mini")
    )
    wf = workflow.chat("{{ 1 + 1 }}", role="user")
    assert isinstance(wf.messages[0], ChatMessage)
    assert wf.messages[0].content == "{{ 1 + 1 }}"

    wf_tpl = workflow.chat("{{ 1 + 1 }}", role="user", as_template=True)
    assert isinstance(wf_tpl.messages[0], agents.MessageTemplate)


# ---------------------------------------------------------------------------
# Stream cleanup — issue #2876
# ---------------------------------------------------------------------------


class ScriptedGenerator(BaseGenerator):
    """Completes the first `fast_first` calls instantly, hangs the rest until cancelled.

    Lets a test observe tasks that are still running when the caller stops listening.
    """

    fast_first: int = Field(default=0)
    started: int = Field(default=0)
    cancelled: int = Field(default=0)

    @override
    async def _call_model(
        self,
        messages: Sequence[ChatMessage],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionResponse:
        self.started += 1
        if self.started <= self.fast_first:
            return self._response("fast")

        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

        return self._response("slow")

    def _response(self, text: str) -> CompletionResponse:
        return CompletionResponse(
            choices=[
                Choice(
                    message=AssistantMessage(content=text),
                    finish_reason="stop",
                    index=0,
                )
            ],
            model="test-model",
        )


def _scripted_workflow(
    fast_first: int,
) -> tuple[agents.ChatWorkflow[Any], ScriptedGenerator]:
    """Build a workflow whose generator hangs, so pending tasks can be observed."""
    generator = ScriptedGenerator(fast_first=fast_first)
    return agents.ChatWorkflow(generator=generator).chat(
        "Hello!", role="user"
    ), generator


def _open_stream(
    stream: AsyncIterator[Chat[Any]],
) -> AsyncGenerator[Chat[Any], None]:
    """Recover the async generator behind a streaming method.

    `stream_many` / `stream_batch` are async generators, but they are annotated as plain
    `AsyncIterator`, so `aclose()` has to be reached through a cast.
    """
    return cast(AsyncGenerator[Chat[Any], None], stream)


async def test_stream_many_still_yields_every_result():
    """Guard for #2876: the cleanup must not change the fully consumed case."""
    workflow, generator = _scripted_workflow(fast_first=3)

    chats = [chat async for chat in workflow.stream_many(3)]

    assert len(chats) == 3
    assert generator.cancelled == 0


async def test_stream_many_stops_pending_tasks_when_consumer_is_cancelled():
    """Regression for #2876: cancelling the consumer must stop the scheduled workflows."""
    workflow, generator = _scripted_workflow(fast_first=0)

    async def consume() -> None:
        async for _ in workflow.stream_many(3):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.3)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert generator.started == 3
    assert generator.cancelled == 3, (
        "workflows scheduled by the stream outlived cancellation"
    )


async def test_stream_many_stops_pending_tasks_when_stream_is_closed():
    """Regression for #2876: closing the stream must stop the workflows still running."""
    workflow, generator = _scripted_workflow(fast_first=1)

    stream = _open_stream(workflow.stream_many(3))
    async for _ in stream:
        break
    await stream.aclose()

    assert generator.started == 3
    assert generator.cancelled == 2, (
        "workflows scheduled by the stream outlived aclose()"
    )


async def test_stream_batch_stops_pending_tasks_when_stream_is_closed():
    """Regression for #2876: `stream_batch` gets the same guarantee as `stream_many`."""
    workflow, generator = _scripted_workflow(fast_first=1)

    stream = _open_stream(workflow.stream_batch(inputs=[{"n": i} for i in range(3)]))
    async for _ in stream:
        break
    await stream.aclose()

    assert generator.started == 3
    assert generator.cancelled == 2, (
        "workflows scheduled by the stream outlived aclose()"
    )
