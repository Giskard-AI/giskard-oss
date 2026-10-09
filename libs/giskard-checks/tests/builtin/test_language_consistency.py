"""Tests for the LanguageConsistency LLM-based check."""

import pytest
from giskard.agents.errors import WorkflowError
from giskard.checks import CheckStatus, Interaction, LanguageConsistency, Trace
from pydantic import ValidationError

from ..testing_utils import (
    InvalidReasonMockJudgeGenerator,
)
from ..testing_utils import (
    MockJudgeGenerator as MockGenerator,
)

# --- Verdict-driven cases -----------------------------------------------
# The linguistic judgment itself happens inside the LLM prompt, not in
# Python, so these tests confirm the check maps a judge verdict to the
# right CheckStatus and forwards the right context — not that the model
# reasons correctly (that was validated manually against a real provider
# before this file was written, see PR description).


async def test_matching_language_passes() -> None:
    """Test that output in the same language as the input passes."""
    generator = MockGenerator(
        passed=True, reason="Response matches the input language."
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(
        inputs="Quelle est la capitale de la France ?",
        outputs="La capitale de la France est Paris.",
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1


async def test_mismatched_language_fails() -> None:
    """Test that output in a different language than the input fails."""
    generator = MockGenerator(
        passed=False,
        reason="Input was in French but the response is entirely in English.",
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(
        inputs="Quelle est la capitale de la France ?",
        outputs="The capital of France is Paris.",
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.FAIL
    assert "English" in (result.details["reason"] or "")


async def test_isolated_technical_terms_do_not_trigger_mismatch() -> None:
    """Test that a bare technical term or proper noun does not flip the verdict.

    The judge is expected to treat "Python" as content, not as a language
    signal, so a French question answered with a French sentence naming a
    proper noun still passes.
    """
    generator = MockGenerator(
        passed=True, reason="Proper noun does not count as a language switch."
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(
        inputs="Quel langage recommandes-tu pour ce projet ?",
        outputs="Je recommande Python pour ce projet.",
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    # The proper noun must reach the judge for the rule to apply.
    assert "Python" in generator.calls[0][0].transcript


async def test_verbatim_quote_does_not_trigger_mismatch() -> None:
    """Test that quoting an error message or exact text in another language passes."""
    generator = MockGenerator(
        passed=True, reason="Quoted error text is not a language switch."
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(
        inputs="Pourquoi ça plante ?",
        outputs="Voici l'erreur retournée : 'File not found'.",
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    # The quoted foreign-language snippet must reach the judge verbatim.
    assert "File not found" in generator.calls[0][0].transcript


async def test_response_without_language_signal_passes_by_default() -> None:
    """Test that a short response carrying no language signal (a number, a URL) passes."""
    generator = MockGenerator(
        passed=True, reason="No language signal in the response; default pass."
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(inputs="Combien ça coûte ?", outputs="42€")
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert "42€" in generator.calls[0][0].transcript


async def test_technical_franglais_passes() -> None:
    """Test the tie-broken jargon case: French grammar, anglicised dev vocabulary."""
    generator = MockGenerator(
        passed=True,
        reason="Content words are English dev jargon but the structure is French; passes per policy.",
    )
    check = LanguageConsistency(generator=generator)
    franglais = "Le check a fail sur le edge case parce que le mock renvoie null."
    interaction = Interaction(
        inputs="Pourquoi le test a échoué ?",
        outputs=franglais,
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert franglais in generator.calls[0][0].transcript


async def test_clause_level_code_switching_fails() -> None:
    """Test that a full clause switching language (not just jargon words) still fails."""
    generator = MockGenerator(
        passed=False,
        reason="A full sentence clause is in English, not just isolated jargon terms.",
    )
    check = LanguageConsistency(generator=generator)
    interaction = Interaction(
        inputs="Peux-tu m'expliquer le bug ?",
        outputs="Bien sûr. The root cause is a race condition in the scheduler.",
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.FAIL
    assert "The root cause is a race condition" in generator.calls[0][0].transcript


async def test_last_turn_input_used_as_reference() -> None:
    """Test that the reference language comes from the LAST user turn, not the first.

    Guards a conversation that starts in one language and switches: the
    check must judge the final output against the final input's language,
    mirroring how Toxicity guards against only looking at the last output
    in isolation (see test_prompt_includes_full_trace_including_prior_turns).
    """
    generator = MockGenerator(passed=True, reason="Matches the last turn's language.")
    check = LanguageConsistency(generator=generator)
    trace = Trace(
        interactions=[
            Interaction(inputs="What's the weather like?", outputs="It's sunny."),
            Interaction(inputs="Et demain ?", outputs="Demain, il pleuvra."),
        ]
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1
    # The resolved reference input must be the last turn, not the first.
    assert result.details["inputs"]["user_input"] == "Et demain ?"


async def test_expected_language_override_takes_precedence() -> None:
    """Test that a fixed expected_language is used instead of the input's language."""
    generator = MockGenerator(
        passed=True, reason="Matches the fixed expected language."
    )
    check = LanguageConsistency(
        generator=generator,
        expected_language="English",
    )
    interaction = Interaction(inputs="Quelle heure est-il ?", outputs="It's 3pm.")
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["expected_language"] == "English"


# --- Mechanical / plumbing cases ----------------------------------------


async def test_prompt_includes_full_history() -> None:
    """The rendered prompt must carry the whole conversation, not just the last turn.

    The reference language can come from an earlier user turn when the last one
    carries no language signal, so the judge needs the full history in context.
    """
    generator = MockGenerator(passed=True, reason="Matches the reference language.")
    check = LanguageConsistency(generator=generator)
    trace = Trace(
        interactions=[
            Interaction(inputs="Explique-moi ce module.", outputs="Bien sur."),
            Interaction(
                inputs="OK", outputs="Ce module normalise les enregistrements."
            ),
        ]
    )
    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    prompt = generator.calls[0][0].transcript
    assert isinstance(prompt, str)
    assert "<HISTORY>" in prompt
    assert "Explique-moi ce module." in prompt


async def test_custom_target_key_and_user_input_key() -> None:
    """Test extraction using custom JSONPath keys for both output and input."""
    generator = MockGenerator(passed=True, reason="Custom keys resolved correctly.")
    check = LanguageConsistency(
        generator=generator,
        target_key="trace.last.outputs.message",
        user_input_key="trace.last.inputs.query",
    )
    interaction = Interaction(
        inputs={"query": "Bonjour"},
        outputs={"message": "Bonjour, comment puis-je vous aider ?"},
    )
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert result.details["inputs"]["output"] == "Bonjour, comment puis-je vous aider ?"
    assert result.details["inputs"]["user_input"] == "Bonjour"


async def test_expected_language_set_user_input_key_not_required() -> None:
    """An unresolvable user_input_key must not error when expected_language is fixed."""
    generator = MockGenerator(
        passed=True, reason="expected_language bypasses input resolution."
    )
    check = LanguageConsistency(
        generator=generator,
        expected_language="French",
        user_input_key="trace.last.inputs.does_not_exist",
    )
    interaction = Interaction(inputs={"query": "hi"}, outputs="Bonjour !")
    result = await check.run(Trace(interactions=[interaction]))

    assert result.status == CheckStatus.PASS
    assert len(generator.calls) == 1


async def test_check_is_serialisable() -> None:
    """Test that the check can be serialised and deserialised via Pydantic."""
    from giskard.agents.generators import Generator
    from giskard.checks.core.check import Check

    check = LanguageConsistency(
        expected_language="French",
        generator=Generator(model="openai/gpt-4o"),
    )
    data = check.model_dump()
    assert data["kind"] == "language_consistency"
    assert data["expected_language"] == "French"

    reconstructed = Check.model_validate(data)
    assert isinstance(reconstructed, LanguageConsistency)
    assert reconstructed.expected_language == "French"


async def test_unresolvable_target_key_returns_error() -> None:
    """An unresolvable target_key (output) always errors, regardless of expected_language."""
    generator = MockGenerator(passed=True, reason="Judge must not run.")
    check = LanguageConsistency(
        generator=generator,
        target_key="trace.last.outputs.does_not_exist",
    )
    trace = Trace(interactions=[Interaction(inputs="Salut", outputs="Hi there.")])

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert len(generator.calls) == 0


async def test_not_does_not_invert_unresolved_target_key_error() -> None:
    """``Not(...)`` must leave the ERROR uninverted, not launder it into PASS."""
    from giskard.checks import Not

    generator = MockGenerator(passed=True, reason="Judge must not run.")
    check = LanguageConsistency(
        generator=generator,
        target_key="trace.last.outputs.does_not_exist",
    )
    trace = Trace(interactions=[Interaction(inputs="Salut", outputs="Hi there.")])

    result = await Not(check=check).run(trace)

    assert result.status == CheckStatus.ERROR
    assert len(generator.calls) == 0


async def test_unresolvable_user_input_key_without_expected_language_returns_error() -> (
    None
):
    """An unresolvable user_input_key errors when there is no expected_language fallback."""
    generator = MockGenerator(passed=True, reason="Judge must not run.")
    check = LanguageConsistency(
        generator=generator,
        user_input_key="trace.last.inputs.does_not_exist",
    )
    trace = Trace(
        interactions=[Interaction(inputs={"query": "hi"}, outputs="Bonjour.")]
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert len(generator.calls) == 0


async def test_blank_reason_raises_workflow_error() -> None:
    """Blank/null judge reasons fail structured-output validation via WorkflowError."""
    check = LanguageConsistency(
        generator=InvalidReasonMockJudgeGenerator(reason=None),
    )
    trace = Trace(interactions=[Interaction(inputs="Bonjour", outputs="Bonjour !")])

    with pytest.raises(WorkflowError) as exc_info:
        _ = await check.run(trace)
    assert isinstance(exc_info.value.exception, ValidationError)
