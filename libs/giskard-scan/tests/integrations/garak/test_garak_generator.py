"""Behavioral tests for the garak TargetGenerator bridge.

The generator adapts garak's synchronous ``Generator._call_model`` API onto a
Giskard ``Target``: it feeds the garak prompt into the target, records the
result in a ``Trace``, and caches that trace per-conversation so multiturn
probes accumulate turns on a single linear trace.

Key contract discovered while writing these tests: the target's first
parameter MUST be named ``inputs`` (an Interact injection key). A parameter
named anything else (e.g. ``prompt``) is rejected by Interact validation.
"""

from uuid import uuid4

import pytest

pytest.importorskip("garak")

from garak.attempt import Conversation, Message, Turn
from giskard.checks import Interaction, Trace
from giskard.scan.integrations.garak._generator import TargetGenerator, _conv_uuid


def _conversation(text: str, uuid: str | None) -> Conversation:
    notes = {"uuid": uuid} if uuid is not None else {}
    return Conversation(
        turns=[Turn(role="user", content=Message(text=text, notes=notes))]
    )


def add_turn(conversation: Conversation, role: str, content: Message) -> Conversation:
    return Conversation(turns=[*conversation.turns, Turn(role=role, content=content)])


def test_play_conversation_handles_multiple_turns() -> None:
    # Two distinct "uuid" concepts flow through this test, do not conflate them:
    #  - the test-side `Interaction.metadata["uuid"]` (managed by stateful_target
    #    below) proves the target sees a continuous trace across turns;
    #  - the generator-side `Message.notes["uuid"]` (set inside _call_model) is
    #    the per-conversation cache key that threads turns together.
    generated_uuids: set[str] = set()
    seen_traces: list[Trace[str, str]] = []

    def stateful_target(inputs: str, trace: Trace[str, str]) -> Interaction[str, str]:
        if trace.last:
            assert "uuid" in trace.last.metadata, "UUID must be preserved in the trace"
            uuid = trace.last.metadata["uuid"]
            assert uuid in generated_uuids, "UUID must have been generated"
        else:
            uuid = str(uuid4())
            generated_uuids.add(uuid)

        seen_traces.append(trace)
        return Interaction(inputs=inputs, outputs=inputs, metadata={"uuid": uuid})

    generator = TargetGenerator(target=stateful_target)

    def play_conversation(messages: list[str]):
        generated_uuid_count = len(generated_uuids)
        conversation = Conversation()
        for message_index, message in enumerate(messages):
            conversation = add_turn(
                conversation, role="user", content=Message(text=message)
            )
            responses = generator._call_model(conversation)
            assert len(generated_uuids) == generated_uuid_count + 1
            assert len(responses) == 1
            response = responses[0]
            assert response is not None
            assert response.text == message
            # The notes uuid is the entire cross-turn threading mechanism;
            # assert it is actually present, not merely that notes exists.
            assert response.notes is not None
            assert "uuid" in response.notes

            assert len(seen_traces) == 1
            trace = seen_traces.pop()
            assert len(trace.interactions) == message_index
            for interaction_index, interaction in enumerate(trace.interactions):
                assert interaction.inputs == messages[interaction_index]
                assert interaction.outputs == messages[interaction_index]

            if len(trace.interactions) > 1:
                assert (
                    len(
                        set(
                            interaction.metadata["uuid"]
                            for interaction in trace.interactions
                        )
                    )
                    == 1
                ), "All interactions should have the same uuid"

            conversation = add_turn(conversation, role="assistant", content=response)

        assert len(conversation.turns) == len(messages) * 2

    play_conversation(["Single turn conversation"])
    play_conversation(["hello", "goodbye"])
    play_conversation(["hello"] * 10)
    assert len(generated_uuids) == 3


def test_play_conversation_handles_structured_outputs() -> None:
    def structured_target(inputs: str) -> int:
        return len(inputs)

    generator = TargetGenerator(target=structured_target)

    def play_conversation(messages: list[str]):
        conversation = Conversation()
        for message in messages:
            conversation = add_turn(
                conversation, role="user", content=Message(text=message)
            )
            responses = generator._call_model(conversation)
            assert len(responses) == 1
            response = responses[0]
            assert response is not None
            assert response.text == str(len(message))
            assert response.notes is not None
            assert "uuid" in response.notes
            conversation = add_turn(conversation, role="assistant", content=response)

        assert len(conversation.turns) == len(messages) * 2

    play_conversation(["Single turn conversation"])
    play_conversation(["hello", "goodbye"])
    play_conversation(["hello"] * 10)


def test_call_model_returns_none_text_when_target_returns_none() -> None:
    def target(inputs: str) -> None:
        return None

    generator = TargetGenerator(target=target)
    messages = generator._call_model(_conversation("hello", uuid="u1"))

    assert len(messages) == 1
    assert messages[0] is not None
    assert messages[0].text is None
    assert messages[0].notes == {"uuid": "u1"}


def test_call_model_invokes_target_and_tags_uuid() -> None:
    def target(inputs: str) -> str:
        return "ECHO:" + inputs

    generator = TargetGenerator(target=target)
    messages = generator._call_model(_conversation("hello", uuid="u1"))

    assert len(messages) == 1
    assert messages[0] is not None
    assert messages[0].text == "ECHO:hello"
    # The uuid is threaded back so the next turn maps to the same trace.
    assert messages[0].notes == {"uuid": "u1"}


def test_call_model_generates_uuid_when_absent() -> None:
    def target(inputs: str) -> str:
        return inputs

    generator = TargetGenerator(target=target)
    messages = generator._call_model(_conversation("hi", uuid=None))

    assert len(messages) == 1
    message = messages[0]
    assert message is not None
    assert message.notes is not None
    assert "uuid" in message.notes
    # A trace is cached under the freshly generated uuid.
    assert list(generator.internal_cache) == [message.notes["uuid"]]


def test_get_trace_pops_cache() -> None:
    def target(inputs: str) -> str:
        return inputs

    generator = TargetGenerator(target=target)
    conversation = _conversation("hi", uuid="u1")
    generator._call_model(conversation)
    assert "u1" in generator.internal_cache

    trace = generator.get_trace(conversation)

    assert trace.last is not None
    # Reading the trace clears it from the cache (pop-on-read).
    assert generator.internal_cache == {}


def test_get_trace_without_uuid_returns_fresh_trace() -> None:
    def target(inputs: str) -> str:
        return inputs

    generator = TargetGenerator(target=target)
    trace = generator.get_trace(_conversation("hi", uuid=None))

    assert trace.last is None


def test_conv_uuid_reads_first_turn_note() -> None:
    assert _conv_uuid(_conversation("x", uuid="abc")) == "abc"
    assert _conv_uuid(_conversation("x", uuid=None)) is None


def test_call_model_skips_target_when_prompt_text_is_none() -> None:
    interaction_counts: list[int] = []

    def stateful_target(inputs: str, trace: Trace[str, str]) -> Interaction[str, str]:
        interaction_counts.append(len(trace.interactions))
        return Interaction(inputs=inputs, outputs=inputs)

    generator = TargetGenerator(target=stateful_target)
    conv_uuid = "thread-1"

    conversation = _conversation("hello", uuid=conv_uuid)
    first = generator._call_model(conversation)
    assert first[0] is not None
    assert first[0].text == "hello"
    assert interaction_counts == [0]

    conversation = add_turn(
        conversation,
        role="assistant",
        content=Message(text="hello", notes={"uuid": conv_uuid}),
    )
    conversation = add_turn(
        conversation, role="user", content=Message(notes={"uuid": conv_uuid})
    )
    second = generator._call_model(conversation)
    assert second[0] is not None
    assert second[0].text is None
    assert interaction_counts == [0]

    conversation = add_turn(conversation, role="assistant", content=second[0])
    conversation = add_turn(
        conversation,
        role="user",
        content=Message(text="goodbye", notes={"uuid": conv_uuid}),
    )
    third = generator._call_model(conversation)
    assert third[0] is not None
    assert third[0].text == "goodbye"
    assert interaction_counts == [0, 1]
