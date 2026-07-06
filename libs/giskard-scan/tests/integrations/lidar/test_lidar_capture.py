import pytest

# Real lidar types; skip the module when lidar is absent (dir has no __init__.py).
pytest.importorskip("lidar")

from giskard.checks import Interaction  # noqa: E402
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
    call_id = response._target_call.call_id
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
        history.append(response.message)
        call_ids.append(response._target_call.call_id)

    assert len(set(call_ids)) == 3  # distinct call per turn
    for i, call_id in enumerate(call_ids):
        assert bridge._by_call_id[call_id].inputs == f"turn {i}"
