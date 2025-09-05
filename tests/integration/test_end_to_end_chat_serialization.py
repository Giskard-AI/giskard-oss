from __future__ import annotations

import json

from counterpoint import Message

from giskard_checks.checks.chat import StringMatchingCheck
from giskard_checks.core import CheckStatus
from giskard_checks.interactions import ChatInteraction
from giskard_checks.testing.testcase import TestCase


async def test_chat_interaction_roundtrip_serialization_and_execution():
    # Prepare a simple chat interaction with one user input and one assistant output
    interaction = ChatInteraction(
        input=[Message(role="user", content="Say hello")],
        output=[Message(role="assistant", content="Hello world!")],
    )

    # One check that should pass, one that should fail
    chk_pass = StringMatchingCheck(
        name="contains_hello",
        content="Hello",
        role="assistant",
        case_sensitive=True,
    )
    chk_fail = StringMatchingCheck(
        name="contains_bye",
        content="bye",
        role="assistant",
        case_sensitive=False,
    )

    tc = TestCase(
        name="tc-chat-serialize",
        interaction=interaction,
        checks=[chk_pass, chk_fail],
    )

    # Run before serialization
    before = await tc.run()
    before_statuses = [r.status for r in before.results]
    assert before_statuses == [CheckStatus.PASS, CheckStatus.FAIL]

    # Serialize and JSON roundtrip to ensure JSON compatibility
    payload = tc.serialize()
    payload_json = json.dumps(payload)
    payload_roundtrip = json.loads(payload_json)

    # Reconstruct
    tc2 = TestCase.deserialize(payload_roundtrip)

    # Structural equivalence
    assert tc2.name == tc.name
    assert tc2.interaction.model_dump() == tc.interaction.model_dump()
    assert [c.kind for c in tc2.checks] == ["string_matching", "string_matching"]

    # Ensure check fields round-trip
    assert isinstance(tc2.checks[0], StringMatchingCheck)
    assert isinstance(tc2.checks[1], StringMatchingCheck)
    assert getattr(tc2.checks[0], "content") == "Hello"
    assert getattr(tc2.checks[0], "role") == "assistant"
    assert getattr(tc2.checks[0], "case_sensitive") is True
    assert getattr(tc2.checks[1], "content") == "bye"
    assert getattr(tc2.checks[1], "role") == "assistant"
    assert getattr(tc2.checks[1], "case_sensitive") is False

    # Run after deserialization and compare outcomes
    after = await tc2.run()
    after_statuses = [r.status for r in after.results]
    assert after_statuses == before_statuses
