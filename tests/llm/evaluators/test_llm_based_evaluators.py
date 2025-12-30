from unittest.mock import Mock

import pandas as pd
import pytest

from giskard.datasets.base import Dataset
from giskard.llm.client import ChatMessage
from giskard.llm.evaluators.base import LLMBasedEvaluator
from giskard.llm.evaluators.plausibility import PlausibilityEvaluator
from giskard.llm.evaluators.requirements import RequirementEvaluator
from giskard.models.base.model_prediction import ModelPredictionResults
from tests.llm.evaluators.utils import make_eval_dataset, make_mock_model


@pytest.mark.parametrize(
    "Evaluator,args,kwargs,additional_metadata",
    [
        (
            LLMBasedEvaluator,
            [],
            {"prompt": "You need to evaluate this model"},
            {},
        ),
        (RequirementEvaluator, [["Requirement to fulfill"]], {}, {}),
        (
            RequirementEvaluator,
            [],
            {"requirement_col": "req"},
            {"req": ["This is the first test requirement", "This is the second test requirement"]},
        ),
        (PlausibilityEvaluator, [], {}, {}),
    ],
)
def test_evaluator_correctly_flags_examples(Evaluator, args, kwargs, additional_metadata):
    eval_dataset = make_eval_dataset()
    eval_dataset.df = eval_dataset.df.assign(**additional_metadata)
    model = make_mock_model()

    client = Mock()
    client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content='{"eval_passed": true}',
        ),
        ChatMessage(
            role="assistant",
            content='{"eval_passed": false, "reason": "For some reason"}',
        ),
    ]

    evaluator = Evaluator(*args, **kwargs, llm_client=client)
    result = evaluator.evaluate(model, eval_dataset)

    assert len(result.success_examples) == 1
    assert len(result.failure_examples) == 1

    assert result.failure_examples[0]["reason"] == "For some reason"
    assert result.failure_examples[0]["sample"]["conversation"][0]["content"] == {
        "question": "What is the airspeed velocity of an unladen swallow?",
        "other": "pass",
    }
    assert (
        result.failure_examples[0]["sample"]["conversation"][1]["content"]
        == "What do you mean? An African or European swallow?"
    )


@pytest.mark.parametrize(
    "Evaluator,args,kwargs",
    [
        (LLMBasedEvaluator, [], {"prompt": "Tell me if the model was any good"}),
        (RequirementEvaluator, [["Requirement to fulfill"]], {}),
        (PlausibilityEvaluator, [], {}),
    ],
)
def test_evaluator_handles_generation_errors(Evaluator, args, kwargs):
    eval_dataset = make_eval_dataset()
    model = make_mock_model()

    client = Mock()
    client.complete.side_effect = [
        ChatMessage(
            role="assistant",
            content='{"eval_passed": true}',
        ),
        ChatMessage(
            role="assistant",
            content='{"model_did_pass_the_test": false}',
        ),
    ]

    evaluator = Evaluator(*args, **kwargs, llm_client=client)

    result = evaluator.evaluate(model, eval_dataset)

    assert len(result.success_examples) == 1
    assert len(result.failure_examples) == 0
    assert len(result.errors) == 1
    assert result.errors[0]["error"] == "Could not parse evaluator output"


def test_evaluator_with_num_retries_fails_on_any_retry():
    """Test that num_retries parameter causes multiple model predictions.

    When num_retries > 1, the evaluator should:
    1. Call model.predict() multiple times for each sample
    2. Mark the sample as failed if ANY retry fails the evaluation

    This tests the fix for issue #2043 where users need to test jailbreak
    scenarios that may only trigger after multiple attempts.
    """
    eval_dataset = Dataset(
        pd.DataFrame({"question": ["Test question"], "other": ["pass"]})
    )

    model = Mock()
    # Model returns different outputs for each retry
    model.predict.side_effect = [
        ModelPredictionResults(prediction=["Safe response 1"]),
        ModelPredictionResults(prediction=["Safe response 2"]),
        ModelPredictionResults(prediction=["Jailbroken response!"]),  # Third retry triggers failure
    ]
    model.feature_names = ["question", "other"]
    model.name = "Mock model"
    model.description = "Test model"

    client = Mock()
    # First two evals pass, third fails
    client.complete.side_effect = [
        ChatMessage(role="assistant", content='{"eval_passed": true}'),
        ChatMessage(role="assistant", content='{"eval_passed": true}'),
        ChatMessage(role="assistant", content='{"eval_passed": false, "reason": "Jailbreak detected"}'),
    ]

    evaluator = LLMBasedEvaluator(
        prompt="Evaluate this conversation",
        llm_client=client,
        num_retries=3,
    )
    result = evaluator.evaluate(model, eval_dataset)

    # Should fail because third retry failed
    assert len(result.failure_examples) == 1
    assert len(result.success_examples) == 0
    assert "retry 3/3" in result.failure_examples[0]["reason"]
    assert "Jailbreak detected" in result.failure_examples[0]["reason"]

    # Model should be called 3 times (until failure)
    assert model.predict.call_count == 3
    # LLM evaluator should be called 3 times
    assert client.complete.call_count == 3


def test_evaluator_with_num_retries_passes_all_retries():
    """Test that if all retries pass, the sample is marked as passed."""
    eval_dataset = Dataset(
        pd.DataFrame({"question": ["Test question"], "other": ["pass"]})
    )

    model = Mock()
    model.predict.side_effect = [
        ModelPredictionResults(prediction=["Response 1"]),
        ModelPredictionResults(prediction=["Response 2"]),
    ]
    model.feature_names = ["question", "other"]
    model.name = "Mock model"
    model.description = "Test model"

    client = Mock()
    client.complete.side_effect = [
        ChatMessage(role="assistant", content='{"eval_passed": true}'),
        ChatMessage(role="assistant", content='{"eval_passed": true}'),
    ]

    evaluator = LLMBasedEvaluator(
        prompt="Evaluate this conversation",
        llm_client=client,
        num_retries=2,
    )
    result = evaluator.evaluate(model, eval_dataset)

    assert len(result.success_examples) == 1
    assert len(result.failure_examples) == 0
    assert model.predict.call_count == 2
    assert client.complete.call_count == 2


def test_evaluator_default_num_retries_is_one():
    """Test that default num_retries is 1 (no retries)."""
    eval_dataset = Dataset(
        pd.DataFrame({"question": ["Test question"], "other": ["pass"]})
    )

    model = Mock()
    model.predict.return_value = ModelPredictionResults(prediction=["Response"])
    model.feature_names = ["question", "other"]
    model.name = "Mock model"
    model.description = "Test model"

    client = Mock()
    client.complete.return_value = ChatMessage(
        role="assistant", content='{"eval_passed": true}'
    )

    evaluator = LLMBasedEvaluator(
        prompt="Evaluate this conversation",
        llm_client=client,
    )
    result = evaluator.evaluate(model, eval_dataset)

    assert len(result.success_examples) == 1
    # Only called once (no retries)
    assert model.predict.call_count == 1
    assert client.complete.call_count == 1
