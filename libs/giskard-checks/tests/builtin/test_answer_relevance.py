"""Unit tests for the AnswerRelevance check.

Tests cover:
- Relevant answer passes
- Off-topic answer fails
- Multi-turn: answer relevant in programming context passes
- Multi-turn: zoology answer to programming question fails (key acceptance criterion)
- Partial relevance (LLM decides)
- Direct question/answer values take priority over trace extraction
- Custom key paths
- Domain context forwarded to template inputs
- Full trace passed as template ``history`` (conversation context for the judge)
- Unresolved ``question_key``/``answer_key`` return ERROR without invoking the judge,
  so a broken key can no longer be silently rescued by the judge inferring the real
  question from conversation history
- ``Not(...)`` leaves those ERRORs uninverted
- ``include_history=False`` omits ``history`` from the template inputs
"""

from typing import Any

from giskard.checks import AnswerRelevance, CheckResult, CheckStatus, Interaction, Trace

from ..testing_utils import MockJudgeGenerator as MockGenerator

_EXPECTED_INPUT_KEYS = frozenset({"question", "answer", "history", "context"})
# ``history`` is omitted from template inputs when include_history=False.
_EXPECTED_INPUT_KEYS_NO_HISTORY = _EXPECTED_INPUT_KEYS - {"history"}


def _assert_answer_relevance_inputs(
    result: CheckResult, *, include_history: bool = True
) -> dict[str, Any]:
    """Assert every AnswerRelevance run records full template inputs."""
    assert "inputs" in result.details
    inputs = result.details["inputs"]
    assert isinstance(inputs, dict)
    expected = (
        _EXPECTED_INPUT_KEYS if include_history else _EXPECTED_INPUT_KEYS_NO_HISTORY
    )
    assert set(inputs.keys()) == expected
    return inputs


def _rendered_prompt(generator: MockGenerator) -> str:
    """Return the single judge prompt that was rendered and sent."""
    assert len(generator.calls) == 1
    return "\n".join(str(message.content) for message in generator.calls[0])


class TestAnswerRelevanceBasic:
    """Basic pass / fail behaviour."""

    async def test_relevant_answer_passes(self):
        """Directly relevant answer should pass."""
        generator = MockGenerator(
            passed=True, reason="Answer directly addresses the question."
        )
        check = AnswerRelevance(
            generator=generator,
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
        )
        result = await check.run(Trace())

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["reason"] == "Answer directly addresses the question."
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is the capital of France?"
        assert inputs["answer"] == "The capital of France is Paris."
        assert inputs["history"] == Trace()
        assert inputs["context"] == ""

    async def test_off_topic_answer_fails(self):
        """Completely off-topic answer should fail."""
        generator = MockGenerator(
            passed=False, reason="Answer is about food, not geography."
        )
        check = AnswerRelevance(
            generator=generator,
            question="What is the capital of France?",
            answer="Lasagna is a delicious Italian dish.",
        )
        result = await check.run(Trace())

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["reason"] == "Answer is about food, not geography."
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is the capital of France?"
        assert inputs["answer"] == "Lasagna is a delicious Italian dish."
        assert inputs["history"] == Trace()
        assert inputs["context"] == ""

    async def test_llm_called_once(self):
        """Exactly one LLM call should be made per check run."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(
            generator=generator,
            question="What is 2 + 2?",
            answer="4.",
        )
        result = await check.run(Trace())

        assert len(generator.calls) == 1
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is 2 + 2?"
        assert inputs["answer"] == "4."
        assert inputs["history"] == Trace()
        assert inputs["context"] == ""


class TestAnswerRelevanceMultiTurn:
    """Multi-turn conversation awareness — core acceptance criteria."""

    async def test_multi_turn_zoology_answer_fails(self):
        """'A snake' answer to 'What's Python?' after programming context should fail.

        This is the canonical acceptance criterion from issue #2338: when the
        conversation history establishes a programming domain, a zoology answer
        to 'What's Python?' must be flagged as not relevant.
        """
        generator = MockGenerator(
            passed=False,
            reason=(
                "Given the prior programming-language context, 'Python' refers to the "
                "language, not the reptile. The answer 'A snake' is off-topic."
            ),
        )
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="What is the best language?", outputs="Python"),
            Interaction(inputs="What's Python?", outputs="A snake."),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        assert inputs["question"] == "What's Python?"
        assert inputs["answer"] == "A snake."
        assert inputs["context"] == ""

    async def test_multi_turn_relevant_programming_answer_passes(self):
        """A correct programming answer in a programming conversation should pass."""
        generator = MockGenerator(
            passed=True,
            reason="Answer correctly describes Python as a programming language.",
        )
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="What is the best language?", outputs="Python"),
            Interaction(
                inputs="What's Python?",
                outputs="Python is a high-level programming language known for its readability.",
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        assert inputs["question"] == "What's Python?"
        assert (
            inputs["answer"]
            == "Python is a high-level programming language known for its readability."
        )
        assert inputs["context"] == ""

    async def test_second_turn_irrelevant_does_not_affect_third_turn_score(self):
        """A prior irrelevant answer must not cause the current relevant answer to fail.

        Issue #2338: 'If a prior message was irrelevant, it should not impact the result.'
        Only the current turn is scored; history is context, not a penalty.
        """
        generator = MockGenerator(
            passed=True,
            reason="The answer correctly identifies Python as both a language and an animal.",
        )
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is the best language?", outputs="You should cook lasagna"
            ),
            Interaction(
                inputs="Is Python a language or an animal?",
                outputs="It's both — Python is a programming language and also a type of snake.",
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        assert inputs["question"] == "Is Python a language or an animal?"
        assert (
            inputs["answer"]
            == "It's both — Python is a programming language and also a type of snake."
        )
        assert inputs["context"] == ""

    async def test_history_in_inputs_is_full_trace(self):
        """Template inputs pass the full trace as history for the judge."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="Turn 1 question", outputs="Turn 1 answer"),
            Interaction(inputs="Turn 2 question", outputs="Turn 2 answer"),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        assert len(inputs["history"].interactions) == 2
        assert inputs["history"].interactions[0].inputs == "Turn 1 question"
        assert inputs["history"].interactions[0].outputs == "Turn 1 answer"
        assert inputs["question"] == "Turn 2 question"
        assert inputs["answer"] == "Turn 2 answer"
        assert inputs["context"] == ""

    async def test_single_turn_history_is_the_trace(self):
        """With one interaction, history is the trace containing that turn."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="Single question", outputs="Single answer"),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        assert len(trace.interactions) == 1
        assert inputs["context"] == ""


class TestAnswerRelevanceInputResolution:
    """Direct values vs. trace extraction."""

    async def test_direct_question_and_answer_used(self):
        """Directly supplied question/answer take priority over trace."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(
            generator=generator,
            question="Direct question",
            answer="Direct answer",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="Trace question", outputs="Trace answer"),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "Direct question"
        assert inputs["answer"] == "Direct answer"
        assert inputs["history"] == trace
        assert inputs["context"] == ""

    async def test_question_and_answer_extracted_from_trace(self):
        """When no direct values given, question/answer extracted from trace."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="What is AI?", outputs="AI is artificial intelligence."),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is AI?"
        assert inputs["answer"] == "AI is artificial intelligence."
        assert inputs["history"] == trace
        assert inputs["context"] == ""

    async def test_custom_keys(self):
        """Custom JSONPath keys should resolve correctly."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(
            generator=generator,
            question_key="trace.interactions[0].inputs.query",
            target_key="trace.interactions[0].outputs.response",
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs={"query": "Custom question"},
                outputs={"response": "Custom answer"},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "Custom question"
        assert inputs["answer"] == "Custom answer"
        assert inputs["history"] == trace
        assert inputs["context"] == ""

    async def test_empty_trace_returns_error_without_judge(self):
        """Empty trace resolves neither key, so the check errors without judging.

        Previously this passed: both keys resolved to ``NoMatch``, were stringified
        into the prompt, and the judge returned PASS anyway.
        """
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)

        result = await check.run(Trace())

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert "question key" in (result.message or "")
        assert "trace.last.inputs" in (result.message or "")
        assert len(generator.calls) == 0


class TestAnswerRelevanceUnresolvedKeys:
    """Unresolved question/answer keys return ERROR without invoking the judge.

    Regression guard for the history leak: ``history`` is the full trace, so before
    this fix a broken ``question_key`` still left the real question visible to the
    judge inside ``<CONVERSATION HISTORY>``. The judge inferred it and returned PASS,
    making the check green regardless of the configured key. Erroring before the
    judge runs removes that path entirely.
    """

    async def test_broken_question_key_errors_without_judge(self):
        """A question_key matching nothing errors instead of passing via history."""
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = AnswerRelevance(
            generator=generator,
            question_key="trace.last.metadata.does_not_exist",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="What is the best language?", outputs="Python"),
            Interaction(inputs="What's Python?", outputs="A snake."),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert "question key" in (result.message or "")
        assert "trace.last.metadata.does_not_exist" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_broken_answer_key_errors_without_judge(self):
        """Same guard for a misconfigured answer_key."""
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = AnswerRelevance(
            generator=generator,
            target_key="trace.last.metadata.does_not_exist",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="What is the best language?", outputs="Python"),
            Interaction(inputs="What's Python?", outputs="A snake."),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert "answer key" in (result.message or "")
        assert "trace.last.metadata.does_not_exist" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_question_key_reported_before_answer_key(self):
        """When both keys miss, the question key is reported first."""
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = AnswerRelevance(
            generator=generator,
            question_key="trace.last.metadata.no_question",
            target_key="trace.last.metadata.no_answer",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="What's Python?", outputs="A snake.")
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert "question key" in (result.message or "")
        assert "trace.last.metadata.no_question" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_directly_provided_values_bypass_broken_keys(self):
        """Direct question/answer take priority, so their keys are never resolved."""
        generator = MockGenerator(passed=True, reason="Relevant.")
        check = AnswerRelevance(
            generator=generator,
            question="Direct question",
            answer="Direct answer",
            question_key="trace.last.metadata.does_not_exist",
            target_key="trace.last.metadata.also_missing",
        )

        result = await check.run(Trace())

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "Direct question"
        assert inputs["answer"] == "Direct answer"

    async def test_not_does_not_invert_unresolved_key_error(self):
        """``Not(...)`` must leave the ERROR uninverted, not launder it into PASS."""
        from giskard.checks import Not

        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = AnswerRelevance(
            generator=generator,
            question_key="trace.last.metadata.does_not_exist",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="What's Python?", outputs="A snake.")
        )

        result = await Not(check=check).run(trace)

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert len(generator.calls) == 0


class TestAnswerRelevanceIncludeHistory:
    """``include_history`` controls whether history reaches the judge."""

    async def test_history_included_by_default(self):
        """History defaults to the full trace and is rendered into the prompt."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="Turn 1 question", outputs="Turn 1 answer"),
            Interaction(inputs="Turn 2 question", outputs="Turn 2 answer"),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["history"] == trace
        prompt = _rendered_prompt(generator)
        assert "<CONVERSATION HISTORY>" in prompt
        assert "Turn 1 question" in prompt

    async def test_include_history_false_omits_history_input(self):
        """Disabling history drops the key from template inputs entirely."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator, include_history=False)
        trace = await Trace.from_interactions(
            Interaction(inputs="Turn 1 question", outputs="Turn 1 answer"),
            Interaction(inputs="Turn 2 question", outputs="Turn 2 answer"),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result, include_history=False)
        assert "history" not in inputs
        assert inputs["question"] == "Turn 2 question"
        assert inputs["answer"] == "Turn 2 answer"

    async def test_include_history_false_drops_prompt_section(self):
        """The rendered prompt omits the history section and prior-turn text."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator, include_history=False)
        trace = await Trace.from_interactions(
            Interaction(inputs="Turn 1 question", outputs="Turn 1 answer"),
            Interaction(inputs="Turn 2 question", outputs="Turn 2 answer"),
        )

        result = await check.run(trace)

        assert result.passed
        prompt = _rendered_prompt(generator)
        assert "<CONVERSATION HISTORY>" not in prompt
        assert "Turn 1 question" not in prompt
        # The current turn still reaches the judge through question/answer.
        assert "Turn 2 question" in prompt
        assert "Turn 2 answer" in prompt

    async def test_include_history_false_still_errors_on_broken_key(self):
        """The unresolved-key guard is independent of the history flag."""
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = AnswerRelevance(
            generator=generator,
            include_history=False,
            question_key="trace.last.metadata.does_not_exist",
        )
        trace = await Trace.from_interactions(
            Interaction(inputs="What's Python?", outputs="A snake.")
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert len(generator.calls) == 0


class TestAnswerRelevancePromptGuardrails:
    """The prompt frames history as supporting context, never a question source."""

    async def test_prompt_forbids_substituting_question_from_history(self):
        """The rendered prompt instructs the judge to score <CURRENT QUESTION> only.

        This is the second layer of the fix: the ERROR guard catches keys that
        resolve to nothing, but a key pointing at a valid-but-wrong field yields a
        real value and passes the guard. Only the prompt stops the judge from
        quietly grading against a question it found in history instead.
        """
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="What's Python?", outputs="A snake.")
        )

        result = await check.run(trace)

        assert result.passed
        prompt = _rendered_prompt(generator)
        assert "<CURRENT QUESTION>" in prompt
        assert "never a source for replacing it" in prompt
        assert "Do **not** search the history for a better-fitting question" in prompt


class TestAnswerRelevanceDomainContext:
    """Optional domain context is forwarded to template inputs."""

    async def test_domain_context_included_in_inputs(self):
        """Supplied domain context must appear in template inputs."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(
            generator=generator,
            question="What is Flask?",
            answer="Flask is a lightweight Python web framework.",
            context="This is a chatbot that answers questions about Python programming.",
        )

        result = await check.run(Trace())

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is Flask?"
        assert inputs["answer"] == "Flask is a lightweight Python web framework."
        assert inputs["history"] == Trace()
        assert inputs["context"] == (
            "This is a chatbot that answers questions about Python programming."
        )

    async def test_no_domain_context_is_empty_string(self):
        """When no domain context is supplied, template input is empty string."""
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = AnswerRelevance(
            generator=generator,
            question="What is Flask?",
            answer="Flask is a web framework.",
        )

        result = await check.run(Trace())

        assert result.passed
        inputs = _assert_answer_relevance_inputs(result)
        assert inputs["question"] == "What is Flask?"
        assert inputs["answer"] == "Flask is a web framework."
        assert inputs["history"] == Trace()
        assert inputs["context"] == ""
