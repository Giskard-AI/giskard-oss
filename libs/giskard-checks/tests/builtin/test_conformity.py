from giskard.checks import CheckStatus, Conformity, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator


async def test_run_returns_success() -> None:
    generator = MockGenerator(passed=True, reason="Rule is followed")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["reason"] == "Rule is followed"

    assert len(generator.calls) == 1
    # The prompt comes from the template file, so we check that the call was made
    assert len(generator.calls[0]) > 0


async def test_run_returns_failure() -> None:
    generator = MockGenerator(passed=False, reason="Rule is violated")
    conformity = Conformity(generator=generator, rule="The response must be polite.")
    result = await conformity.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.details["reason"] == "Rule is violated"

    assert len(generator.calls) == 1


async def test_trace_in_result_details_inputs() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    conformity = Conformity(generator=generator, rule="Test rule")
    interaction = Interaction(
        inputs={"query": "What is AI?"}, outputs={"response": "AI is..."}
    )
    trace = Trace(interactions=[interaction])
    result = await conformity.run(trace)

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["rule"] == "Test rule"
    assert "trace" in result.details["inputs"]

    stored_trace = result.details["inputs"]["trace"]
    assert stored_trace is trace


async def test_conformity_renders_with_spy_metadata_without_raising() -> None:
    """``WithSpy`` metadata must stay JSON-serialisable (gap G-08b).

    ``WithSpy`` previously stored raw ``unittest.mock._Call`` objects in
    interaction metadata. The ``Conformity`` judge template renders
    ``{{ trace | fence }}`` through ``Trace.model_dump(mode="json")``, which
    raised ``TypeError: '_Call' object is not an instance of 'SchemaSerializer'``
    and surfaced as ``WorkflowError: Step processing failed``. WithSpy now stores
    plain ``{"args", "kwargs"}`` dicts, so a judge paired with ``WithSpy`` renders
    without raising.
    """
    from unittest.mock import MagicMock

    from giskard.checks import Interact, WithSpy

    target = "app.apply_tool"

    # Drive WithSpy._patch_interaction exactly as generate() does, so the test
    # exercises the real serialisation path rather than hand-built metadata.
    with_spy = WithSpy(
        interaction_generator=Interact(inputs="n/a", outputs="ok"),
        target=target,
    )
    spy = MagicMock()
    spy("user@example.com", "I want to apply")
    patched = with_spy._patch_interaction(
        Interaction(inputs="I want to apply", outputs="Application saved"),
        spy,
    )
    trace = Trace(interactions=[patched])

    generator = MockGenerator(passed=True, reason="The tool was called.")
    conformity = Conformity(
        generator=generator,
        rule="The application has been saved and the tool was called.",
    )
    result = await conformity.run(trace)

    # The judge rendered the fenced trace (including WithSpy metadata) and
    # returned the canned verdict, instead of raising during rendering.
    assert result.status == CheckStatus.PASS, result.details
    rendered = "\n".join(m.text for m in generator.calls[0] if m.text)
    assert "call_args" in rendered
    assert "user@example.com" in rendered

    # Metadata is plain JSON-serialisable dicts (no unittest.mock._Call objects).
    dumped = trace.model_dump(mode="json")
    spy_meta = dumped["interactions"][0]["metadata"][target]
    assert spy_meta["call_args"] == {
        "args": ["user@example.com", "I want to apply"],
        "kwargs": {},
    }
    assert spy_meta["call_count"] == 1
    assert spy_meta["call_args_list"] == [
        {"args": ["user@example.com", "I want to apply"], "kwargs": {}}
    ]
    assert spy_meta["mock_calls"][0]["args"] == [
        "user@example.com",
        "I want to apply",
    ]


async def test_empty_trace_in_result_details_inputs() -> None:
    generator = MockGenerator(passed=True, reason="Mock reason.")
    conformity = Conformity(generator=generator, rule="Test rule")
    trace = Trace()
    result = await conformity.run(trace)

    assert result.status == CheckStatus.PASS
    assert "inputs" in result.details
    assert result.details["inputs"]["rule"] == "Test rule"
    stored_trace = result.details["inputs"]["trace"]
    assert stored_trace is trace
