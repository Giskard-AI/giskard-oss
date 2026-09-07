"""Tests for the deepteam model_callback bridge and its uuid trace cache."""

import pytest

pytest.importorskip("deepteam")

from deepteam.test_case import RTTurn
from giskard.scan.integrations.deepteam._bridge import ScanTargetCallback


async def test_single_turn_drives_target_and_returns_assistant_rtturn():
    calls = []

    def target(inputs: str) -> str:
        calls.append(inputs)
        return f"reply to: {inputs}"

    cb = ScanTargetCallback(target=target)
    out = await cb("hello")

    assert isinstance(out, RTTurn)
    assert out.role == "assistant"
    assert out.content == "reply to: hello"
    assert calls == ["hello"]
    # A uuid was stamped so multi-turn can thread it.
    assert out.metadata and cb.METADATA_UUID_KEY in out.metadata


async def test_multi_turn_accumulates_one_typed_trace_across_calls():
    def target(inputs: str) -> str:
        return f"reply to: {inputs}"

    cb = ScanTargetCallback(target=target)

    # Turn 1: no prior turns.
    first = await cb("turn-1", turns=[])
    uuid = first.metadata[cb.METADATA_UUID_KEY]

    # Simulate DeepTeam threading: prior turns include our stamped assistant turn.
    turns = [
        RTTurn(role="user", content="turn-1"),
        first,
        RTTurn(role="user", content="turn-2"),
    ]
    second = await cb("turn-2", turns=turns)

    # Same conversation -> same uuid -> one accumulating trace with both turns.
    assert second.metadata[cb.METADATA_UUID_KEY] == uuid
    trace = cb.traces[uuid]
    assert len(trace.interactions) == 2
    assert trace.last is not None
    assert trace.last.outputs == "reply to: turn-2"
    # Structured inputs preserved (not collapsed to a bare str turn record).
    assert trace.interactions[0].inputs is not None


async def test_new_conversation_gets_fresh_uuid_and_trace():
    def target(inputs: str) -> str:
        return "ok"

    cb = ScanTargetCallback(target=target)
    a = await cb("a", turns=[])
    b = await cb("b", turns=[])  # no prior assistant turn -> new conversation
    assert a.metadata[cb.METADATA_UUID_KEY] != b.metadata[cb.METADATA_UUID_KEY]
    assert len(cb.traces) == 2
