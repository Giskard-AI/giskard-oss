from counterpoint import Message
from pydantic import BaseModel

from giskard_checks.checks import StringMatchingCheck
from giskard_checks.checks.fn import EqualityCheck, FnCheck
from giskard_checks.interactions import ChatInteraction, StructuredInteraction
from giskard_checks.testing import TestCase


class CheckResult(BaseModel):
    passed: bool
    reason: str
    score: float


class LLMAsAJudge:
    def evaluate(
        self, description: str, question: str, answer: str, reference_answer: str
    ) -> CheckResult:
        if answer in reference_answer:
            return CheckResult(
                passed=True, reason=f"Answer is in the reference answer", score=1.0
            )
        else:
            return CheckResult(
                passed=False, reason=f"Answer is not in the reference answer", score=0.0
            )


async def test_llm_as_a_judge():
    llm_as_a_judge = LLMAsAJudge()

    input = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input)

    testcase = TestCase(
        name="test_llm_as_a_judge",
        interaction=StructuredInteraction(
            input=input,
            output=result,
        ),
        checks=[
            EqualityCheck(
                expected=True,
                key="output.passed",
            ),
            EqualityCheck(
                expected=True,
                key="output.passed",
            ),
        ],
    )

    result = await testcase.run()

    assert result.passed


async def test_chat_bot_context_relevance():
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
    chat_interaction = ChatInteraction(
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

    testcase = TestCase(
        name="test_chat_bot_context_relevance",
        interaction=chat_interaction,
        checks=[
            StringMatchingCheck(
                content="Paris",
                key="metadata.context.documents[*].content",
                match_all=True,
            )
        ],
    )

    result = await testcase.run()

    assert not result.passed

    testcase = TestCase(
        name="test_chat_bot_context_relevance",
        interaction=chat_interaction,
        checks=[
            StringMatchingCheck(
                content="Paris",
                key="metadata.context.documents[*].content",
                match_all=False,
            )
        ],
    )

    result = await testcase.run()

    assert result.passed
