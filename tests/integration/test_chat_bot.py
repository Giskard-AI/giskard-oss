"""Integration tests for chat bot scenarios using the giskard_checks testing framework."""

from pydantic import BaseModel

from giskard_checks.checks import StringMatchingCheck
from giskard_checks.interactions import StructuredInteraction
from giskard_checks.testing import TestCase
from giskard_checks.testing._samples.message import Message


class ChatResponse(BaseModel):
    """Mock chat response for testing."""

    message: str
    confidence: float
    sources: list[str]


class ChatBot:
    """Mock chat bot implementation for testing."""

    def respond(
        self, conversation: list[Message], context: dict[str, str] | None = None
    ) -> list[Message]:
        """Generate a response to a conversation."""
        _ = context  # Acknowledge parameter to avoid linting warning
        if not conversation:
            return [
                Message(
                    role="assistant",
                    content="I need a message to respond to.",
                )
            ]

        last_message = str(conversation[-1].content).lower()

        if "capital" in last_message and "france" in last_message:
            return [
                Message(
                    role="assistant",
                    content="Paris is the capital of France and has a population of 2.1 million.",
                )
            ]
        elif "population" in last_message and "paris" in last_message:
            return [
                Message(
                    role="assistant",
                    content="Paris has a population of approximately 2.1 million people.",
                )
            ]
        elif "hello" in last_message:
            return [
                Message(
                    role="assistant",
                    content="Hello! How can I help you today?",
                )
            ]
        else:
            return [
                Message(
                    role="assistant",
                    content="I'm not sure how to answer that question.",
                )
            ]


async def test_chat_bot_basic_conversation():
    """Test basic chat bot conversation flow."""
    chat_bot = ChatBot()

    conversation = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="Paris"),
        Message(role="user", content="What is the population of Paris?"),
    ]

    answer = chat_bot.respond(conversation)

    testcase = TestCase(
        name="test_chat_bot_basic_conversation",
        interaction=StructuredInteraction(
            input=conversation,
            output=answer,
        ),
        checks=[
            StringMatchingCheck(
                content="Paris",
                key="output[*].content",
            ),
            StringMatchingCheck(
                content="2.1 million",
                key="output[*].content",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_chat_bot_context_relevance():
    """Test chat bot with context relevance checks."""
    conversation = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="Paris"),
        Message(role="user", content="What is the population of Paris?"),
    ]

    answer = [
        Message(
            role="assistant",
            content="Paris is the capital of France and has a population of 2.1 million.",
        ),
    ]

    chat_interaction = StructuredInteraction(
        input=conversation,
        output=answer,
        metadata={
            "context": {
                "documents": [
                    {
                        "id": "1",
                        "content": "Paris is the capital of France and has a population of 2.1 million.",
                    },
                    {
                        "id": "2",
                        "content": "France is a country in Europe.",
                    },
                ]
            },
        },
    )

    # Test that not all documents contain "Paris" (should fail with evaluation_mode="all")
    testcase = TestCase(
        name="test_chat_bot_context_relevance_match_all",
        interaction=chat_interaction,
        checks=[
            StringMatchingCheck(
                content="Paris",
                key="metadata.context.documents[*].content",
                evaluation_mode="all",
            )
        ],
    )

    result = await testcase.run()
    assert not result.passed

    # Test that at least one document contains "Paris" (should pass with evaluation_mode="any")
    testcase = TestCase(
        name="test_chat_bot_context_relevance_match_any",
        interaction=chat_interaction,
        checks=[
            StringMatchingCheck(
                content="Paris",
                key="metadata.context.documents[*].content",
                evaluation_mode="any",
            )
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_chat_bot_conversation_history():
    """Test chat bot with conversation history validation."""
    conversation = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="Paris"),
        Message(role="user", content="What is the population of Paris?"),
    ]

    answer = [
        Message(
            role="assistant",
            content="Paris has a population of approximately 2.1 million people.",
        ),
    ]

    testcase = TestCase(
        name="test_chat_bot_conversation_history",
        interaction=StructuredInteraction(
            input=conversation,
            output=answer,
        ),
        checks=[
            # Check that the conversation contains the expected flow
            StringMatchingCheck(
                content="capital of France",
                key="input[*].content",
            ),
            StringMatchingCheck(
                content="population of Paris",
                key="input[*].content",
            ),
            # Check that the response is relevant to the last question
            StringMatchingCheck(
                content="population",
                key="output[*].content",
            ),
            StringMatchingCheck(
                content="2.1 million",
                key="output[*].content",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_chat_bot_multiple_turns():
    """Test chat bot with multiple conversation turns."""
    conversation = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hello! How can I help you today?"),
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="Paris"),
        Message(role="user", content="What is the population of Paris?"),
    ]

    answer = [
        Message(
            role="assistant",
            content="Paris has a population of approximately 2.1 million people.",
        ),
    ]

    testcase = TestCase(
        name="test_chat_bot_multiple_turns",
        interaction=StructuredInteraction(
            input=conversation,
            output=answer,
        ),
        checks=[
            # Verify all messages have roles
            StringMatchingCheck(
                content="user",
                key="input[*].role",
            ),
            StringMatchingCheck(
                content="assistant",
                key="input[*].role",
            ),
            # Verify response quality
            StringMatchingCheck(
                content="population",
                key="output[*].content",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_chat_bot_edge_cases():
    """Test chat bot with edge cases and boundary conditions."""
    # Test with empty conversation
    empty_conversation: list[Message] = []

    # Test with very long conversation
    long_conversation = [
        Message(role="user", content=f"Message {i}")
        for i in range(10)  # Reduced from 100 for testing
    ]

    # Test with special characters
    special_conversation = [
        Message(
            role="user",
            content="What's the capital of France? (with special chars: @#$%)",
        ),
    ]

    test_cases = [
        ("empty_conversation", empty_conversation),
        ("long_conversation", long_conversation),
        ("special_conversation", special_conversation),
    ]

    for test_name, conversation in test_cases:
        # Mock response for edge cases
        answer = [
            Message(
                role="assistant",
                content="I understand your question.",
            ),
        ]

        testcase = TestCase(
            name=f"test_chat_bot_edge_cases_{test_name}",
            interaction=StructuredInteraction(
                input=conversation,
                output=answer,
            ),
            checks=[
                # Basic check that we get a response
                StringMatchingCheck(
                    content="assistant",
                    key="output[*].role",
                ),
                # Check that response contains expected content
                StringMatchingCheck(
                    content="understand",
                    key="output[*].content",
                ),
            ],
        )

        result = await testcase.run()
        assert result.passed


async def test_chat_bot_none_evaluation_mode():
    """Test chat bot with 'none' evaluation mode to ensure certain content is NOT present."""
    conversation = [
        Message(role="user", content="What is the capital of France?"),
    ]

    answer = [
        Message(
            role="assistant",
            content="Paris is the capital of France.",
        ),
    ]

    testcase = TestCase(
        name="test_chat_bot_none_evaluation_mode",
        interaction=StructuredInteraction(
            input=conversation,
            output=answer,
        ),
        checks=[
            # Test that "London" is NOT present in the response
            StringMatchingCheck(
                content="London",
                key="output[*].content",
                evaluation_mode="none",
            ),
            # Test that "Berlin" is NOT present in the response
            StringMatchingCheck(
                content="Berlin",
                key="output[*].content",
                evaluation_mode="none",
            ),
            # Test that "Paris" IS present (using default "any" mode)
            StringMatchingCheck(
                content="Paris",
                key="output[*].content",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_chat_bot_none_evaluation_mode_failure():
    """Test chat bot with 'none' evaluation mode that should fail when content IS present."""
    conversation = [
        Message(role="user", content="What is the capital of France?"),
    ]

    answer = [
        Message(
            role="assistant",
            content="Paris is the capital of France.",
        ),
    ]

    testcase = TestCase(
        name="test_chat_bot_none_evaluation_mode_failure",
        interaction=StructuredInteraction(
            input=conversation,
            output=answer,
        ),
        checks=[
            # This should fail because "Paris" IS present in the response
            StringMatchingCheck(
                content="Paris",
                key="output[*].content",
                evaluation_mode="none",
            ),
        ],
    )

    result = await testcase.run()
    assert not result.passed
