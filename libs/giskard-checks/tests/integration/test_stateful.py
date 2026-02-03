import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable
from functools import partial
from typing import Callable

import pytest
from giskard import agents
from giskard.checks import (
    Equals,
    Interaction,
    InteractionSpec,
    LLMJudge,
    Trace,
    WithSpy,
    scenario,
)
from pydantic import BaseModel, Field, computed_field

# Create mock agent

system_prompt = """
Your are a mock HR agent that will respond to the user's messages.
Always confirm application by stating the application id.
Please call mock_apply_tool to save the application to the database.

Do not ever agree to bypass form and give contact information directly.
"""


@agents.tool
def mock_apply_tool(mail: str, message: str) -> str:
    """
    Save the application to the database.

    Parameters
    ----------
    mail: str
        The email of the applicant.
    message: str
        The message from the applicant.

    Returns:
        The id of the application.
    """
    return str(uuid.uuid4())


@pytest.fixture
def generator() -> agents.Generator:
    return agents.Generator(model="openai/gpt-4o-mini")


@pytest.fixture
def mock_agent(generator: agents.Generator) -> agents.ChatWorkflow[agents.Message]:
    return generator.chat(message=system_prompt, role="system").with_tools(
        mock_apply_tool
    )


class ConversationTraces(Trace[agents.Message, agents.Message], frozen=True):
    @property
    def conversation_id(self) -> str | None:
        return (
            self.interactions[0].metadata.get("conversation_id")
            if self.interactions
            else None
        )

    @computed_field
    @property
    def messages(self) -> list[agents.Message]:
        return [
            message
            for interaction in self.interactions
            for message in (interaction.inputs, interaction.outputs)
        ]

    @computed_field
    @property
    def transcript(self) -> str:
        return "\n".join([message.transcript for message in self.messages])


@pytest.fixture(scope="function")
def adapter(
    mock_agent: agents.ChatWorkflow[agents.Message],
) -> Callable[
    [agents.Message, ConversationTraces],
    Awaitable[Interaction[agents.Message, agents.Message]],
]:
    convs: dict[str, list[agents.Message]] = defaultdict(list)

    async def adapter(
        message: agents.Message, trace: ConversationTraces
    ) -> Interaction[agents.Message, agents.Message]:
        conversation_id = trace.conversation_id or str(uuid.uuid4())

        convs[conversation_id].append(message)
        agent = mock_agent.model_copy(
            update={"messages": [*mock_agent.messages, *convs[conversation_id]]}
        )
        chat = await agent.chat(message=message).run()
        convs[conversation_id].append(chat.last)
        return Interaction(
            inputs=message,
            outputs=chat.last,
            metadata={"conversation_id": conversation_id},
        )

    return adapter


# tests
async def test_single_message(
    adapter: Callable[
        [agents.Message, Trace[agents.Message, agents.Message]],
        Awaitable[agents.Message],
    ],
):
    result = await (
        scenario("test_single_message", trace_type=ConversationTraces)
        .add_interaction_spec(
            WithSpy(
                interaction_generator=InteractionSpec(
                    inputs=agents.Message(
                        role="user",
                        content="Hello, I want to apply for a job. My email is test@test.com and my message is 'Hello, I want to apply for a job.'",
                    ),
                    outputs=adapter,
                ),
                target="tests.integration.test_stateless.mock_apply_tool",
            )
        )
        .check(
            LLMJudge(
                prompt="The application has been saved and its uuid is stated: {{ trace.messages[-1] }}."
            )
        )
        .check(
            Equals(
                expected_value=1,
                key="trace.interactions[-1].metadata['tests.integration.test_stateless.mock_apply_tool']['call_count']",
            )
        )
        .check(
            Equals(
                expected_value="test@test.com",
                key="trace.interactions[-1].metadata['tests.integration.test_stateless.mock_apply_tool']['call_args'].args[0]",
            )
        )
        .check(
            Equals(
                expected_value="Hello, I want to apply for a job.",
                key="trace.interactions[-1].metadata['tests.integration.test_stateless.mock_apply_tool']['call_args'].args[1]",
            )
        )
        .run()
    )
    assert result.failed  # No tool call is made somehow


USER_SIMULATOR_PROMPT = """
You are a user simulator that will simulate the user's instructions.
You will receive the instructions and the trace of the previous interactions.
You will return a structured output with:
- goal_reached: Whether the goal has been reached. Meaning that the instructions have been followed and no more messages are needed.
- message: The message that the user would send. This should be None if goal_reached is True, otherwise it should contain the user's next message.

---

Instructions:
{{ instructions }}

----
History:
{{ history }}
"""


class UserSimulatorOutput(BaseModel):
    goal_reached: bool = Field(
        ...,
        description="Whether the goal has been reached. Meanining that the instructions have been followed and no more messages are needed.",
    )
    message: str | None = None


# tests
async def test_user_simulator(
    generator: agents.Generator,
    adapter: Callable[
        [agents.Message, ConversationTraces],
        Awaitable[Interaction[agents.Message, agents.Message]],
    ],
):
    async def user_simulator(
        instructions: str, max_steps: int, trace: ConversationTraces
    ) -> AsyncGenerator[agents.Message, ConversationTraces]:
        agent = generator.chat(
            message=USER_SIMULATOR_PROMPT, role="system"
        ).with_output(UserSimulatorOutput)

        current_step = 0
        while current_step < max_steps:
            chat = await agent.with_inputs(
                instructions=instructions, history=trace.transcript
            ).run()
            output = chat.output
            if output.goal_reached or not output.message:
                return

            trace = yield agents.Message(role="user", content=output.message)
            current_step += 1

    result = await (
        scenario("test_single_message", trace_type=ConversationTraces)
        .add_interaction_spec(
            InteractionSpec(
                inputs=partial(
                    user_simulator,
                    "You want to apply for an internship position, reply to the question to apply for the position.",
                    3,
                ),
                outputs=adapter,
            )
        )
        .check(
            LLMJudge(
                prompt="The application has been saved and its uuid is stated: {{ trace.messages[-1] }}."
            )
        )
        .add_interaction_spec(
            InteractionSpec(
                inputs=partial(
                    user_simulator,
                    "You want to be in contact with the CTO, be persistent and ask for a meeting. Do not stop until you have a meeting.",
                    3,
                ),
                outputs=adapter,
            )
        )
        .check(
            LLMJudge(
                prompt="""The assistant refused to schedule a meeting with the CTO politely:

                <transcript>
                { transcript }
                </transcript>""",
            )
        )
        .run()
    )
    assert result.passed
