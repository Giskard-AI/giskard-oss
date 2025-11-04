"""Integration tests for LLM-as-a-judge scenarios using the giskard.checks testing framework."""

from giskard.checks.checks import EqualityCheck, StringMatchingCheck
from giskard.checks.generators import Interaction
from giskard.checks.testing import TestCase
from pydantic import BaseModel


class CheckResult(BaseModel):
    """Mock check result for testing."""

    passed: bool
    reason: str
    score: float


class LLMAsAJudge:
    """Mock LLM-as-a-judge implementation for testing."""

    def evaluate(
        self, description: str, question: str, answer: str, reference_answer: str
    ) -> CheckResult:
        """Evaluate an answer against a reference answer."""
        # Use all parameters to avoid linting warnings
        _ = description, question  # Acknowledge parameters are used in context

        # Handle empty answers as failures
        if not answer.strip():
            return CheckResult(
                passed=False, reason=f"Answer is not in the reference answer", score=0.0
            )

        # Check if answer is contained in reference answer or vice versa
        if answer in reference_answer or reference_answer in answer:
            return CheckResult(
                passed=True, reason=f"Answer is in the reference answer", score=1.0
            )
        else:
            return CheckResult(
                passed=False, reason=f"Answer is not in the reference answer", score=0.0
            )


async def test_llm_judge_basic_success():
    """Test basic LLM judge evaluation with successful case."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_basic_success",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            EqualityCheck(
                expected=True,
                key="outputs.passed",
            ),
            EqualityCheck(
                expected=1.0,
                key="outputs.score",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_basic_failure():
    """Test basic LLM judge evaluation with failure case."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "London",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_basic_failure",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            EqualityCheck(
                expected=False,
                key="outputs.passed",
            ),
            EqualityCheck(
                expected=0.0,
                key="outputs.score",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_partial_match():
    """Test LLM judge with partial answer matching."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris, the capital",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_partial_match",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            EqualityCheck(
                expected=True,
                key="outputs.passed",
            ),
            StringMatchingCheck(
                content="Answer is in the reference answer",
                key="outputs.reason",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_multiple_checks_mixed_results():
    """Test LLM judge with multiple checks where some pass and some fail."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_multiple_checks_mixed_results",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            EqualityCheck(
                expected=True,
                key="outputs.passed",
            ),
            EqualityCheck(
                expected=1.0,
                key="outputs.score",
            ),
            # This check will fail, making the overall test fail
            EqualityCheck(
                expected=False,
                key="outputs.passed",
            ),
        ],
    )

    result = await testcase.run()
    assert not result.passed


async def test_llm_judge_with_metadata():
    """Test LLM judge with metadata in the interaction."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_with_metadata",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
            metadata={
                "model": "gpt-4",
                "temperature": 0.1,
                "evaluation_criteria": ["accuracy", "completeness"],
            },
        ),
        checks=[
            EqualityCheck(
                expected=True,
                key="outputs.passed",
            ),
            StringMatchingCheck(
                content="gpt-4",
                key="metadata.model",
            ),
            EqualityCheck(
                expected=0.1,
                key="metadata.temperature",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_error_handling():
    """Test LLM judge with error handling scenarios."""
    llm_as_a_judge = LLMAsAJudge()

    # Test with empty answer
    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_error_handling",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            EqualityCheck(
                expected=False,
                key="outputs.passed",
            ),
            StringMatchingCheck(
                content="Answer is not in the reference answer",
                key="outputs.reason",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_complex_evaluation():
    """Test LLM judge with complex evaluation criteria."""
    llm_as_a_judge = LLMAsAJudge()

    # Test multiple questions and answers
    test_cases = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "reference_answer": "Paris",
            "expected_passed": True,
        },
        {
            "question": "What is the capital of Germany?",
            "answer": "Berlin",
            "reference_answer": "Berlin",
            "expected_passed": True,
        },
        {
            "question": "What is the capital of Spain?",
            "answer": "Madrid",
            "reference_answer": "Madrid",
            "expected_passed": True,
        },
    ]

    for i, test_case in enumerate(test_cases):
        input_data = {
            "description": "You are a judge that evaluates the answer to a question.",
            "question": str(test_case["question"]),
            "answer": str(test_case["answer"]),
            "reference_answer": str(test_case["reference_answer"]),
        }

        result = llm_as_a_judge.evaluate(**input_data)

        testcase = TestCase(
            name=f"test_llm_judge_complex_evaluation_{i}",
            interaction=Interaction(
                inputs=input_data,
                outputs=result,
            ),
            checks=[
                EqualityCheck(
                    expected=test_case["expected_passed"],
                    key="outputs.passed",
                ),
            ],
        )

        result = await testcase.run()
        assert result.passed


async def test_llm_judge_none_evaluation_mode():
    """Test LLM judge with 'none' evaluation mode to ensure certain content is NOT present."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_none_evaluation_mode",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            # Test that "error" is NOT present in the reason
            StringMatchingCheck(
                content="error",
                key="outputs.reason",
                evaluation_mode="none",
            ),
            # Test that "failed" is NOT present in the reason
            StringMatchingCheck(
                content="failed",
                key="outputs.reason",
                evaluation_mode="none",
            ),
            # Test that "Answer is in the reference answer" IS present
            StringMatchingCheck(
                content="Answer is in the reference answer",
                key="outputs.reason",
            ),
        ],
    )

    result = await testcase.run()
    assert result.passed


async def test_llm_judge_none_evaluation_mode_failure():
    """Test LLM judge with 'none' evaluation mode that should fail when content IS present."""
    llm_as_a_judge = LLMAsAJudge()

    input_data = {
        "description": "You are a judge that evaluates the answer to a question.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
    }

    result = llm_as_a_judge.evaluate(**input_data)

    testcase = TestCase(
        name="test_llm_judge_none_evaluation_mode_failure",
        interaction=Interaction(
            inputs=input_data,
            outputs=result,
        ),
        checks=[
            # This should fail because "Answer" IS present in the reason
            StringMatchingCheck(
                content="Answer",
                key="outputs.reason",
                evaluation_mode="none",
            ),
        ],
    )

    result = await testcase.run()
    assert not result.passed
