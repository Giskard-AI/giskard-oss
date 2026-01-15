from pathlib import Path

from giskard import agents
from giskard.agents.chat import Chat
from giskard.agents.generators.litellm_generator import LiteLLMGenerator
from giskard.agents.templates.prompts_manager import PromptsManager
from pydantic import BaseModel


async def test_single_run(generator):
    workflow = agents.ChatWorkflow(generator=generator)

    chat = await (
        workflow.chat("Your name is TestBot.", role="system")
        .chat("What is your name? Answer in one word.", role="user")
        .run()
    )

    assert "testbot" in chat.last.content.lower()


async def test_run_many(generator):
    """Test that the workflow runs correctly."""

    workflow = agents.ChatWorkflow(generator=generator)

    chats = await workflow.chat("Hello!", role="user").run_many(n=3)

    assert len(chats) == 3


async def test_run_batch(generator):
    """Test that the workflow runs correctly."""

    workflow = agents.ChatWorkflow(generator=generator)

    chats = await workflow.chat("Hello {{ n }}!", role="user").run_batch(
        inputs=[{"n": i} for i in range(3)]
    )

    assert chats[0].context.inputs["n"] == 0
    assert chats[1].context.inputs["n"] == 1
    assert chats[2].context.inputs["n"] == 2

    assert chats[0].messages[0].content == "Hello 0!"
    assert chats[1].messages[0].content == "Hello 1!"
    assert chats[2].messages[0].content == "Hello 2!"

    assert len(chats) == 3


async def test_stream_many(generator):
    workflow = agents.ChatWorkflow(generator=generator).chat("Hello!", role="user")

    chats = []
    async for chat in workflow.stream_many(3):
        assert isinstance(chat, Chat)
        chats.append(chat)

    assert len(chats) == 3


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


async def test_workflow_with_mixed_templates(generator: LiteLLMGenerator):
    workflow = agents.ChatWorkflow(
        generator=generator,
        prompt_manager=PromptsManager(
            default_prompts_path=Path(__file__).parent / "data" / "prompts"
        ),
    )

    chat = (
        await workflow.template("multi_message.j2")
        .chat("{{ score }}!", role="assistant")
        .chat("Well done {{ name }}!", role="user")
        .with_inputs(
            name="TestBot",
            theory="Normandy is actually the center of the universe.",
            score=100,
        )
        .run()
    )

    assert len(chat.messages) == 5

    assert chat.messages[0].role == "system"
    assert (
        "You are an impartial evaluator of scientific theories."
        in chat.messages[0].content
    )

    assert chat.messages[1].role == "user"
    assert (
        "Normandy is actually the center of the universe." in chat.messages[1].content
    )

    assert chat.messages[2].role == "assistant"
    assert "100" in chat.messages[2].content

    assert chat.messages[3].role == "user"
    assert "Well done TestBot!" in chat.messages[3].content

    assert chat.messages[4].role == "assistant"


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
