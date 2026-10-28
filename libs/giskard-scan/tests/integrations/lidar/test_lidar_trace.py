import pytest

# Match the garak tests: real lidar types, skip the module when lidar is absent.
pytest.importorskip("lidar")

from giskard.scan.integrations.lidar._adapter import _trace_from_messages  # noqa: E402
from lidar.giskard_compat import make_message  # noqa: E402


async def test_trace_pairs_user_and_assistant():
    messages = [
        make_message(role="user", content="hello"),
        make_message(role="assistant", content="hi there"),
        make_message(role="user", content="again"),
        make_message(role="assistant", content="sure"),
    ]
    trace = await _trace_from_messages(messages)
    interactions = list(trace.interactions)
    assert len(interactions) == 2
    assert interactions[0].inputs == "hello"
    assert interactions[0].outputs == "hi there"
    assert interactions[1].inputs == "again"
    assert interactions[1].outputs == "sure"


async def test_trace_skips_system_messages():
    messages = [
        make_message(role="system", content="you are a bot"),
        make_message(role="user", content="q"),
        make_message(role="assistant", content="a"),
    ]
    trace = await _trace_from_messages(messages)
    interactions = list(trace.interactions)
    assert len(interactions) == 1
    assert interactions[0].inputs == "q"
    assert interactions[0].outputs == "a"


async def test_trace_trailing_user_has_no_output():
    messages = [make_message(role="user", content="dangling")]
    trace = await _trace_from_messages(messages)
    interactions = list(trace.interactions)
    assert len(interactions) == 1
    assert interactions[0].inputs == "dangling"
    assert interactions[0].outputs is None


from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

from giskard.checks import Interaction  # noqa: E402
from giskard.scan.integrations.lidar._adapter import LidarScanAdapter  # noqa: E402


@dataclass
class _FakeTargetCall:
    call_id: str


@dataclass
class _FakeAttempt:
    target_calls: list[Any] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)


class _FakeBridge:
    def __init__(self, by_call_id):
        self._by_call_id = by_call_id


async def test_trace_from_target_calls_joins_in_order():
    # Two calls; the bridge captured a typed Interaction for each. The rebuilt
    # trace must be those exact interactions, in target_calls order.
    i0 = Interaction(inputs="q0", outputs="a0")
    i1 = Interaction(inputs="q1", outputs="a1")
    bridge = _FakeBridge({"c0": i0, "c1": i1})
    attempt = _FakeAttempt(
        target_calls=[_FakeTargetCall("c0"), _FakeTargetCall("c1")],
        messages=[],  # deliberately empty: proves we did NOT use messages
    )
    trace = await LidarScanAdapter()._trace_from_target_calls(attempt, bridge)
    assert [it.inputs for it in trace.interactions] == ["q0", "q1"]
    assert [it.outputs for it in trace.interactions] == ["a0", "a1"]


async def test_trace_from_target_calls_recovers_structure_when_messages_hidden():
    # A probe may show a sanitized transcript in messages while the real call is
    # in target_calls; the join must recover the real structured input.
    real = Interaction(inputs="the real payload", outputs="ok")
    bridge = _FakeBridge({"c0": real})
    from lidar.giskard_compat import make_message

    attempt = _FakeAttempt(
        target_calls=[_FakeTargetCall("c0")],
        messages=[make_message(role="assistant", content="[redacted]")],
    )
    trace = await LidarScanAdapter()._trace_from_target_calls(attempt, bridge)
    assert len(trace.interactions) == 1
    assert trace.interactions[0].inputs == "the real payload"


async def test_trace_from_target_calls_falls_back_to_messages_when_no_calls():
    # No target_calls (probe errored before reaching the target): fall back to
    # reconstructing from messages so partial transcripts still display.
    from lidar.giskard_compat import make_message

    bridge = _FakeBridge({})
    attempt = _FakeAttempt(
        target_calls=[],
        messages=[
            make_message(role="user", content="hello"),
            make_message(role="assistant", content="hi"),
        ],
    )
    trace = await LidarScanAdapter()._trace_from_target_calls(attempt, bridge)
    assert len(trace.interactions) == 1
    assert trace.interactions[0].inputs == "hello"
    assert trace.interactions[0].outputs == "hi"


async def test_trace_from_target_calls_skips_unknown_call_ids():
    # A call_id the bridge never captured (should not happen, but must not crash):
    # skip it rather than inserting a None interaction.
    bridge = _FakeBridge({"c0": Interaction(inputs="q0", outputs="a0")})
    attempt = _FakeAttempt(
        target_calls=[_FakeTargetCall("c0"), _FakeTargetCall("missing")],
        messages=[],
    )
    trace = await LidarScanAdapter()._trace_from_target_calls(attempt, bridge)
    assert [it.inputs for it in trace.interactions] == ["q0"]
