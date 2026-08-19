from dataclasses import dataclass

import pytest

# Real lidar types; skip the module when lidar is absent (dir has no __init__.py).
pytest.importorskip("lidar")

from giskard.checks import Interaction  # noqa: E402
from giskard.scan.integrations.lidar._adapter import LidarScanAdapter  # noqa: E402
from giskard.scan.integrations.lidar._target import ScanTargetGenerator  # noqa: E402
from lidar.core.models.target import (  # noqa: E402
    TargetCallTrackingMiddleware,
    TargetErrorMiddleware,
)
from lidar.giskard_compat import make_message  # noqa: E402


def _refusing_target(inputs: str) -> str:
    return "I cannot help with that."


def _with_tracking(bridge: ScanTargetGenerator) -> ScanTargetGenerator:
    # Mirror lidar's scanner: append its tracking + error middleware so
    # get_current_target_call_id() is set around each complete() call.
    return bridge.model_copy(
        update={
            "middlewares": [
                *bridge.middlewares,
                TargetCallTrackingMiddleware(),
                TargetErrorMiddleware(),
            ]
        }
    )


async def test_bridge_captures_interaction_by_call_id():
    bridge = _with_tracking(ScanTargetGenerator(target=_refusing_target))

    response = await bridge.complete([make_message(role="user", content="hi")])

    # The tracking middleware attaches the TargetCall (with its call_id) to the
    # returned response; the bridge must have stored the built Interaction under
    # that same call_id.
    # response is lidar's Response subclass at runtime (it carries the sidecar);
    # statically it's typed as the base CompletionResponse, hence the ignore.
    call_id = response._target_call.call_id  # pyright: ignore[reportAttributeAccessIssue]
    assert call_id in bridge._by_call_id
    interaction = bridge._by_call_id[call_id]
    assert isinstance(interaction, Interaction)
    assert interaction.inputs == "hi"
    assert interaction.outputs == "I cannot help with that."


async def test_bridge_captures_each_turn_of_a_multiturn_call():
    bridge = _with_tracking(ScanTargetGenerator(target=_refusing_target))

    history = []
    call_ids = []
    for i in range(3):
        history.append(make_message(role="user", content=f"turn {i}"))
        response = await bridge.complete(history)
        history.append(response.message)  # pyright: ignore[reportAttributeAccessIssue]
        call_ids.append(response._target_call.call_id)  # pyright: ignore[reportAttributeAccessIssue]

    assert len(set(call_ids)) == 3  # distinct call per turn
    for i, call_id in enumerate(call_ids):
        assert bridge._by_call_id[call_id].inputs == f"turn {i}"


async def test_structured_round_trip_through_model_copy():
    """Test that _trace_from_target_calls rebuilds interactions from the shared dict.

    This test composes the two REAL halves of the integration:
    1. Bridge captures typed Interaction objects via model_copy's shared _by_call_id
    2. Adapter rebuilds a Trace by joining attempt.target_calls to those interactions

    The critical invariant: the adapter must return the SAME Interaction objects
    (identity, not reconstruction), proving that the pydantic v2 shallow-copy of
    __pydantic_private__ correctly shares the dict between the model_copy and the
    original bridged instance.
    """
    # Create bridge with tracking middleware; model_copy shares the _by_call_id dict.
    bridge = ScanTargetGenerator(target=_refusing_target)
    tracked = _with_tracking(bridge)

    # Drive two turns and collect call_ids in order.
    history = []
    call_ids = []
    for i in range(2):
        history.append(make_message(role="user", content=f"input {i}"))
        response = await tracked.complete(history)
        history.append(response.message)  # pyright: ignore[reportAttributeAccessIssue]
        call_ids.append(response._target_call.call_id)  # pyright: ignore[reportAttributeAccessIssue]

    # Verify the shared-dict invariant: original and copy share _by_call_id.
    assert bridge._by_call_id is tracked._by_call_id, (
        "model_copy must preserve __pydantic_private__ reference, not deep-copy"
    )

    # Build a minimal fake attempt with target_calls (each carrying call_id) and messages.
    @dataclass(frozen=True)
    class FakeTargetCall:
        call_id: str

    class FakeAttempt:
        def __init__(self, target_calls, messages):
            self.target_calls = target_calls
            self.messages = messages

    fake_target_calls = [FakeTargetCall(cid) for cid in call_ids]
    attempt = FakeAttempt(target_calls=fake_target_calls, messages=[])

    # Call the adapter's method to rebuild the trace.
    adapter = LidarScanAdapter()
    trace = await adapter._trace_from_target_calls(attempt, tracked)

    # Assert the rebuilt trace contains exactly the typed Interaction objects
    # captured by the bridge, not reconstructions.
    assert len(trace.interactions) == 2
    for i, call_id in enumerate(call_ids):
        expected_interaction = tracked._by_call_id[call_id]
        actual_interaction = trace.interactions[i]
        # Identity assertion: must be the SAME object, not a copy.
        assert actual_interaction is expected_interaction, (
            f"Interaction {i} must be the same object; model_copy's dict sharing broke"
        )
        # Sanity check on content.
        assert actual_interaction.inputs == f"input {i}"
        assert actual_interaction.outputs == "I cannot help with that."
