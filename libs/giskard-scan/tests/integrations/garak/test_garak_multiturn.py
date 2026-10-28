"""Real multiturn test: drive a genuine garak ``IterativeProbe`` end to end.

The other functional test (test_garak_e2e.py) runs a *single-turn* shipped probe.
This one closes the multiturn gap: it exercises garak's real iterative BFS loop
(``IterativeProbe.probe`` -> ``_execute_all`` -> ``_call_model`` -> next-turn
attempts) against a stub target, so we prove that:

  * a multiturn conversation is threaded through ``TargetGenerator``'s uuid cache,
    accumulating turns on a single ``Trace`` across calls;
  * ``target_mode="singleturn"`` filters iterative probes out.

garak ships no ``IterativeProbe`` subclass we can rely on, so we define a minimal
2-turn one here. Everything else — the probe base class, ``Conversation``/``Turn``/
``Message``, the ``always.Fail`` detector, config loading — is real garak.

Marked ``functional``: runs whenever garak is installed, skipped otherwise.
"""

import pytest
from giskard.checks import SuiteResult
from giskard.scan.integrations.garak._adapter import garak_available

pytestmark = pytest.mark.functional


if garak_available():
    import garak.attempt as ga
    from garak.probes.base import IterativeProbe

    class _TinyMultiturnProbe(IterativeProbe):
        """Minimal real iterative probe: an opening prompt then one follow-up."""

        lang = "en"
        primary_detector = "always.Fail"
        end_condition = "detector"
        max_calls_per_conv = 2
        follow_prompt_cap = False
        goal = "exercise the multiturn bridge"
        active = True

        def _create_init_attempts(self):
            return [self._create_attempt("turn-1 prompt")]

        def _generate_next_attempts(self, last_attempt):
            conversation = last_attempt.conversations[0]
            user_turns = sum(1 for turn in conversation.turns if turn.role == "user")
            if user_turns >= 2:
                return []
            next_conversation = ga.Conversation(turns=list(conversation.turns))
            next_conversation.turns.append(
                ga.Turn("user", ga.Message("turn-2 follow-up", lang=self.lang or "en"))
            )
            return [self._create_attempt(next_conversation)]


def _patch_probe(monkeypatch: pytest.MonkeyPatch, probe_cls):
    """Return our probe instead of garak's real catalog; return the live adapter.

    Patches and returns the adapter from the live ``_adapter`` module, like
    test_garak_run.py's ``_patch_resolvers`` (see conftest.py ``block_garak_import``
    for why the live module matters).

    The probe is built *inside* the patched resolver, not before: garak's
    ``Probe.__init__`` copies ``generations``/``parallel_attempts`` off the global
    config at instantiation, and the adapter loads that config (``_configure_garak``)
    just before calling ``_resolve_probes``. Instantiating earlier would miss the
    config and raise AttributeError mid-probe.
    """
    from giskard.scan.integrations.garak import _adapter

    monkeypatch.setattr(
        _adapter, "_resolve_probes", lambda probes_arg: ([probe_cls()], [])
    )
    return _adapter.GarakScanAdapter


@pytest.mark.skipif(not garak_available(), reason="garak is not installed")
async def test_real_iterative_probe_accumulates_multiturn_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def target(inputs: str) -> str:
        calls.append(inputs)
        return f"reply to: {inputs}"

    adapter_cls = _patch_probe(monkeypatch, _TinyMultiturnProbe)
    result = await adapter_cls().run(target=target, target_mode="multiturn")

    assert isinstance(result, SuiteResult)

    # garak's iterative loop drove both turns against the target, in order.
    assert calls == ["turn-1 prompt", "turn-2 follow-up"]

    # The completed root attempt (#1) carries the full 2-turn conversation,
    # so its final_trace accumulated both interactions on one Trace — proof the
    # uuid cache threaded the conversation across separate _call_model calls.
    root = next(s for s in result.results if s.scenario_name.endswith("#1"))
    assert len(root.final_trace.interactions) == 2
    assert root.final_trace.last is not None
    assert root.final_trace.last.outputs == "reply to: turn-2 follow-up"


@pytest.mark.skipif(not garak_available(), reason="garak is not installed")
async def test_singleturn_mode_skips_iterative_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(inputs: str) -> str:
        return "unused"

    adapter_cls = _patch_probe(monkeypatch, _TinyMultiturnProbe)
    result = await adapter_cls().run(target=target, target_mode="singleturn")

    # The only probe is iterative, so singleturn mode leaves nothing to run.
    assert result.results == []
