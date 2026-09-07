"""End-to-end test for GarakScanAdapter.run with a stubbed probe.

Rather than drive a real garak probe (slow, downloads weights, non-deterministic),
we inject a fake Probe and a fake Detector by patching the two resolver helpers.
This exercises our orchestration only: the asyncio.TaskGroup fan-out, result
gathering after the group exits, and the SuiteResult assembly.
"""

import pytest

pytest.importorskip("garak")

from garak.attempt import Attempt, Conversation, Message, Turn
from giskard.checks import SuiteResult
from giskard.scan.integrations import third_party_scan


def _patch_resolvers(monkeypatch: pytest.MonkeyPatch, probes, detectors):
    """Patch the resolvers on the live _adapter module and return its adapter.

    ``run()`` looks up ``_resolve_probes`` / ``_resolve_detectors`` as module
    globals, so we patch — and return the adapter from — the exact live module.
    See conftest.py ``block_garak_import`` for why a top-level import could bind a
    stale module and silently run the real garak catalog.

    ``detectors`` is a list of ``(label, detector)`` pairs, matching
    ``_resolve_detectors`` output.
    """
    from giskard.scan.integrations.garak import _adapter

    monkeypatch.setattr(_adapter, "_resolve_probes", lambda probes_arg: (probes, []))
    monkeypatch.setattr(
        _adapter, "_resolve_detectors", lambda probe, loop, *_: (detectors, [])
    )
    return _adapter.GarakScanAdapter


class _FakeDetector:
    detectorname = "garak.detectors.fake.Detector"
    hit_desc = "hit"
    pass_desc = "pass"

    def __init__(self, score: float) -> None:
        self._score = score

    def detect(self, attempt: Attempt) -> list[float]:
        # One score per conversation in the attempt.
        return [self._score for _ in attempt.conversations]


class _TruncatedDetector:
    detectorname = "garak.detectors.fake.TruncatedDetector"
    hit_desc = "hit"
    pass_desc = "pass"

    def detect(self, attempt: Attempt) -> list[float]:
        return [0.9]


class _OverProducingDetector:
    """Returns more scores than the attempt has conversations."""

    detectorname = "garak.detectors.fake.OverProducingDetector"
    hit_desc = "hit"
    pass_desc = "pass"

    def detect(self, attempt: Attempt) -> list[float]:
        return [0.9] * (len(attempt.conversations) + 2)


class _FakeProbe:
    probename = "fake.Probe"
    tags = ["owasp:llm01", "quality:Security:PromptStability"]

    def __init__(self, attempts: list[Attempt]) -> None:
        self._attempts = attempts

    def probe(self, generator: object) -> list[Attempt]:
        return self._attempts


class _FailingProbe:
    probename = "fake.FailingProbe"
    tags: list[str] = []

    def probe(self, generator: object) -> list[Attempt]:
        raise RuntimeError("probe blew up")


class _CacheCheckingProbe:
    probename = "fake.CacheProbe"
    tags: list[str] = []

    def __init__(self, probe_id: str) -> None:
        self.probe_id = probe_id

    def probe(self, generator: object) -> list[Attempt]:
        from giskard.scan.integrations.garak._generator import TargetGenerator

        assert isinstance(generator, TargetGenerator)
        assert generator.internal_cache == {}, (
            f"probe {self.probe_id} inherited a dirty generator cache"
        )
        return [_attempt(self.probe_id)]


def _attempt(uuid: str) -> Attempt:
    conversation = Conversation(
        turns=[Turn(role="user", content=Message(text="hi", notes={"uuid": uuid}))]
    )
    return Attempt(prompt=conversation)


def _attempt_with_conversations(conversations: list[Conversation]) -> Attempt:
    attempt = Attempt()
    attempt.conversations = conversations
    return attempt


@pytest.fixture
def target():
    def _target(inputs: str) -> str:
        return "ECHO:" + inputs

    return _target


async def test_run_produces_suite_result_from_fake_probe(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    probe = _FakeProbe([_attempt("a1"), _attempt("a2")])
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[probe],
        detectors=[("fake.Detector", _FakeDetector(score=0.9))],
    )

    result = await adapter_cls().run(target=target)

    assert isinstance(result, SuiteResult)
    # Two attempts, each with one conversation -> two scenarios.
    assert len(result.results) == 2
    assert result.duration_ms >= 0
    for scenario in result.results:
        assert scenario.scenario_name.startswith("fake.Probe")
        assert scenario.tags == probe.tags


async def test_run_failure_score_marks_check_failed(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[_FakeProbe([_attempt("a1")])],
        detectors=[("fake.Detector", _FakeDetector(score=0.9))],  # > 0.5 -> failure
    )

    result = await adapter_cls().run(target=target)

    check_results = result.results[0].steps[0].results
    assert len(check_results) == 1
    assert check_results[0].failed
    assert check_results[0].details["check_name"] == "fake.Detector"


async def test_run_emits_error_for_probe_exception(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[_FakeProbe([_attempt("a1")]), _FailingProbe()],
        detectors=[("fake.Detector", _FakeDetector(score=0.1))],
    )

    result = await adapter_cls().run(target=target)

    assert len(result.results) == 2
    by_name = {s.scenario_name: s for s in result.results}
    assert any(n.startswith("fake.Probe") for n in by_name)
    error_key = next(n for n in by_name if "FailingProbe" in n)
    assert by_name[error_key].errored
    check = by_name[error_key].steps[0].results[0]
    assert check.errored
    assert "probe blew up" in (check.message or "")
    # A crashing probe must not inflate pass_rate via an empty suite.
    assert result.pass_rate is not None and result.pass_rate < 1.0
    assert result.errored_count == 1


async def test_run_uses_separate_generator_per_probe(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[_CacheCheckingProbe("p1"), _CacheCheckingProbe("p2")],
        detectors=[("fake.Detector", _FakeDetector(score=0.1))],
    )

    result = await adapter_cls().run(target=target)

    assert len(result.results) == 2


async def test_run_with_no_probes_returns_empty_suite(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    adapter_cls = _patch_resolvers(
        monkeypatch, probes=[], detectors=[("fake.Detector", _FakeDetector(score=0.1))]
    )

    result = await adapter_cls().run(target=target)

    assert result.results == []


async def test_run_rejects_unexpected_kwargs(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    # A typo'd option (e.g. probe vs probes) must raise, not be silently dropped.
    adapter_cls = _patch_resolvers(
        monkeypatch, probes=[], detectors=[_FakeDetector(score=0.1)]
    )

    with pytest.raises(TypeError, match="probe"):
        await adapter_cls().run(target=target, probe=["x"])


async def test_third_party_scan_routes_to_garak_adapter(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    """The public entry point resolves tool="garak" to GarakScanAdapter.run."""
    _patch_resolvers(
        monkeypatch,
        probes=[_FakeProbe([_attempt("a1")])],
        detectors=[("fake.Detector", _FakeDetector(score=0.9))],
    )

    result = await third_party_scan(target, tool="garak", description="A test agent")

    assert isinstance(result, SuiteResult)
    assert len(result.results) == 1
    assert result.results[0].scenario_name.startswith("fake.Probe")


async def test_third_party_scan_rejects_unknown_tool(target) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        await third_party_scan(  # pyright: ignore[reportCallIssue]
            target,
            tool="nope",  # pyright: ignore[reportArgumentType]
            description="A test agent",
        )


async def test_third_party_scan_requires_description(target) -> None:
    with pytest.raises(TypeError):
        await third_party_scan(target, tool="garak")  # pyright: ignore[reportCallIssue]


async def test_run_marks_missing_detector_scores_as_skip(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    conversations = [
        Conversation(
            turns=[Turn(role="user", content=Message(text="hi", notes={"uuid": "a"}))]
        ),
        Conversation(
            turns=[Turn(role="user", content=Message(text="bye", notes={"uuid": "b"}))]
        ),
    ]
    probe = _FakeProbe([_attempt_with_conversations(conversations)])
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[probe],
        detectors=[("fake.TruncatedDetector", _TruncatedDetector())],
    )

    result = await adapter_cls().run(target=target)

    assert len(result.results) == 2
    assert result.results[0].steps[0].results[0].failed
    assert result.results[1].steps[0].results[0].skipped
    assert (
        result.results[1].steps[0].results[0].message
        == "no detector score for this conversation"
    )


async def test_run_warns_and_drops_extra_detector_scores(
    monkeypatch: pytest.MonkeyPatch, target, caplog
) -> None:
    conversations = [
        Conversation(
            turns=[Turn(role="user", content=Message(text="hi", notes={"uuid": "a"}))]
        ),
    ]
    probe = _FakeProbe([_attempt_with_conversations(conversations)])
    adapter_cls = _patch_resolvers(
        monkeypatch,
        probes=[probe],
        detectors=[("fake.OverProducingDetector", _OverProducingDetector())],
    )

    with caplog.at_level("WARNING"):
        result = await adapter_cls().run(target=target)

    # Extra scores don't create phantom scenarios: one conversation, one result.
    assert len(result.results) == 1
    assert result.results[0].steps[0].results[0].failed
    assert "returned 3 scores for 1 conversations" in caplog.text


class _LoopBlockingProbe:
    """Probe that mimics the real _call_model: from its worker thread it schedules
    a coroutine on the scan loop and BLOCKS on the result. That coroutine itself
    needs a worker thread (asyncio.to_thread), exactly like a structured target's
    LLM call. If probes shared asyncio.to_thread's default pool, enough of them
    would fill it, all block on the loop, and starve the threads their own
    coroutines need -> deadlock. The dedicated per-probe pool prevents this.
    """

    tags: list[str] = []

    def __init__(self, name: str, loop) -> None:
        self.probename = name
        self._loop = loop

    def probe(self, generator: object) -> list[Attempt]:
        import asyncio

        async def _needs_a_thread() -> str:
            # This inner work needs a free pool thread to complete.
            return await asyncio.to_thread(lambda: "ok")

        fut = asyncio.run_coroutine_threadsafe(_needs_a_thread(), self._loop)
        fut.result()  # block this worker thread on the loop, like _call_model
        return [_attempt(self.probename)]


async def test_run_does_not_deadlock_when_probes_exceed_thread_pool(
    monkeypatch: pytest.MonkeyPatch, target
) -> None:
    """Probe workers block on the loop; their coroutines need pool threads too.

    With N probes > the shared default executor (min(32, cpu+4)), running them on
    asyncio.to_thread makes every worker block on run_coroutine_threadsafe while
    the coroutines can't get a thread to finish -> deadlock. A dedicated pool
    sized to the probe count keeps the run finishing regardless of probe count.
    """
    import asyncio

    from giskard.scan.integrations.garak import _adapter

    loop = asyncio.get_running_loop()
    # Well above the default pool; ~40 mirrors a full probe set.
    probes = [_LoopBlockingProbe(f"p{i}", loop) for i in range(40)]

    monkeypatch.setattr(
        _adapter, "_resolve_probes", lambda probes_arg: (probes_arg or [], [])
    )
    monkeypatch.setattr(
        _adapter,
        "_resolve_detectors",
        lambda probe, loop, *_: ([("fake.Detector", _FakeDetector(score=0.1))], []),
    )

    result = await asyncio.wait_for(
        _adapter.GarakScanAdapter().run(target=target, probes=probes),
        timeout=30,
    )

    assert len(result.results) == 40
