import pytest
from giskard.checks.checks import Groundedness
from giskard.checks.core import Check, Interaction

from tests.test_utils.mock_generator import MockGenerator


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Grounded"),
        (False, "Not grounded"),
    ],
)
async def test_groundedness_pass_fail_roundtrip(passed: bool, reason: str):
    interaction = Interaction(
        inputs="This is a test question.", outputs="This is a test response."
    )

    answer_value = "The Eiffel Tower is in Paris."
    context_list = ["The Eiffel Tower is located in Paris, France."]

    check = Groundedness(
        answer=answer_value,
        context=context_list,
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["answer"] == answer_value
    assert res1.details["inputs"]["context"] == str(context_list)
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Groundedness)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["answer"] == answer_value
    assert res2.details["inputs"]["context"] == str(context_list)
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Grounded"),
        (False, "Not grounded"),
    ],
)
async def test_groundedness_pass_fail_roundtrip_default_keys(passed: bool, reason: str):
    answer_value = "The Eiffel Tower is in Paris."
    context_list = ["The Eiffel Tower is located in Paris, France."]
    interaction = Interaction(
        inputs="This is a test question.",
        outputs={"answer": answer_value},
        metadata={
            "context": [{"name": "context", "value": ctx} for ctx in context_list]
        },
    )

    check = Groundedness(
        answer_key="$.outputs.answer",
        context_key="$.metadata.context[*].value",
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["answer"] == answer_value
    assert res1.details["inputs"]["context"] == str(context_list)
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Groundedness)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["answer"] == answer_value
    assert res2.details["inputs"]["context"] == str(context_list)
    assert res2.message == reason


@pytest.mark.parametrize(
    "passed, reason",
    [
        (True, "Grounded"),
        (False, "Not grounded"),
    ],
)
async def test_groundedness_pass_fail_roundtrip_custom_keys(passed: bool, reason: str):
    answer_value = "The Eiffel Tower is in Paris."
    context_list = ["The Eiffel Tower is located in Paris, France."]
    interaction = Interaction(
        inputs="This is a test question.",
        outputs=answer_value,
        metadata={"context": context_list},
    )

    check = Groundedness(
        generator=MockGenerator(
            outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
        ),
    )

    res1 = await check.run(interaction)
    assert res1.passed == passed
    assert res1.details["reason"] == reason
    assert res1.details["inputs"]["answer"] == answer_value
    assert res1.details["inputs"]["context"] == str([context_list])
    assert res1.message == reason

    serialized = check.model_dump()
    deserialized = Check.model_validate(serialized)
    assert isinstance(deserialized, Groundedness)
    # Note: generator field is excluded from serialization

    # Set the same generator for the deserialized check
    deserialized.generator = MockGenerator(
        outputs=f'{{"passed": {str(passed).lower()}, "reason": "{reason}"}}'
    )

    res2 = await deserialized.run(interaction)
    assert res2.passed == passed
    assert res2.details["reason"] == reason
    assert res2.details["inputs"]["answer"] == answer_value
    assert res2.details["inputs"]["context"] == str([context_list])
    assert res2.message == reason
