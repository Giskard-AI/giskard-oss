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
