"""Unit tests for the ContextRelevance check.

Tests cover:
- Relevant retrieved chunk passes; irrelevant chunk fails
- Context given as ``list[str]`` and as a single ``str``
- Multi-turn: the "How do I install it?" disambiguation scenario
- History resolution (prior turns only) and omission on a single-turn trace
- Direct ``query`` value takes priority over trace extraction
- Domain context forwarded to template inputs
- Unresolved ``query_key``/``context_key`` return ERROR without invoking the judge
- ``Not(...)`` leaves those ERRORs uninverted
"""

from typing import Any

from giskard.checks import (
    CheckResult,
    CheckStatus,
    ContextRelevance,
    Interaction,
    Not,
    Trace,
)

from ..testing_utils import MockJudgeGenerator as MockGenerator

_EXPECTED_INPUT_KEYS = frozenset({"query", "context", "history", "domain_context"})


def _assert_inputs(result: CheckResult) -> dict[str, Any]:
    """Assert every ContextRelevance run records full template inputs."""
    assert "inputs" in result.details
    inputs = result.details["inputs"]
    assert isinstance(inputs, dict)
    assert set(inputs.keys()) == _EXPECTED_INPUT_KEYS
    return inputs


def _rendered_prompt(generator: MockGenerator) -> str:
    """Return the single judge prompt that was rendered and sent."""
    assert len(generator.calls) == 1
    return "\n".join(str(message.content) for message in generator.calls[0])


class TestContextRelevanceBasic:
    """Standard RAG pass / fail behaviour."""

    async def test_relevant_chunk_passes(self):
        generator = MockGenerator(
            passed=True, reason="Context contains the definition of Python."
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="Python is a language.",
                metadata={"context": ["Python is a high-level programming language."]},
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert result.details["reason"] == "Context contains the definition of Python."
        inputs = _assert_inputs(result)
        assert inputs["query"] == "What is Python?"
        assert inputs["context"] == "Python is a high-level programming language."
        assert inputs["history"] == ""
        assert inputs["domain_context"] == ""

    async def test_irrelevant_chunk_fails(self):
        generator = MockGenerator(
            passed=False, reason="Context is about cooking, not the query."
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="Python is a language.",
                metadata={"context": ["Lasagna is baked in layers with béchamel."]},
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.FAIL
        assert result.failed
        assert result.details["reason"] == "Context is about cooking, not the query."
        inputs = _assert_inputs(result)
        assert inputs["context"] == "Lasagna is baked in layers with béchamel."

    async def test_llm_called_once(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is 2 + 2?",
                outputs="4.",
                metadata={"context": "2 + 2 is 4."},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        assert len(generator.calls) == 1


class TestContextRelevanceContextShapes:
    """Context resolves from both ``list[str]`` and ``str``."""

    async def test_list_context_joined_into_single_block(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": ["First chunk.", "Second chunk."]},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["context"] == "First chunk.\nSecond chunk."
        prompt = _rendered_prompt(generator)
        assert "First chunk." in prompt
        assert "Second chunk." in prompt
        # The Python list repr must not leak into the prompt.
        assert "['First chunk." not in prompt

    async def test_str_context_passed_through(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": "Python is a programming language."},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["context"] == "Python is a programming language."


class TestContextRelevanceMultiTurn:
    """Multi-turn disambiguation — core acceptance criterion of issue #2339."""

    async def test_install_it_follow_up_passes(self):
        """'How do I install it?' with install docs for the earlier subject passes."""
        generator = MockGenerator(
            passed=True,
            reason="Context contains installation instructions for the requested tool.",
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Giskard checks?",
                outputs="Giskard checks is an evaluation library.",
                metadata={"context": ["Giskard checks is an evaluation library."]},
            ),
            Interaction(
                inputs="How do I install it?",
                outputs="Run pip install giskard-checks.",
                metadata={"context": ["To install, run `pip install giskard-checks`."]},
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["query"] == "How do I install it?"
        assert inputs["context"] == "To install, run `pip install giskard-checks`."
        # Only prior turns are passed as history; the evaluated turn is not.
        assert "What is Giskard checks?" in inputs["history"]
        assert "How do I install it?" not in inputs["history"]
        prompt = _rendered_prompt(generator)
        assert "<CONVERSATION HISTORY>" in prompt
        assert "What is Giskard checks?" in prompt

    async def test_prior_irrelevant_turn_does_not_affect_verdict(self):
        """A prior irrelevant exchange must not sink the current retrieval."""
        generator = MockGenerator(
            passed=True, reason="Context answers the current query."
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is the best language?",
                outputs="You should try to cook lasagna",
                metadata={"context": ["Lasagna recipe."]},
            ),
            Interaction(
                inputs="Is Python a language or an animal?",
                outputs="It's both",
                metadata={"context": ["Python is both a language and a snake."]},
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["query"] == "Is Python a language or an animal?"
        assert inputs["context"] == "Python is both a language and a snake."

    async def test_single_turn_trace_has_no_history_section(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": "A language."},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["history"] == ""
        assert "<CONVERSATION HISTORY>" not in _rendered_prompt(generator)


class TestContextRelevanceInputResolution:
    """Direct values, custom keys, and domain context."""

    async def test_direct_query_takes_priority(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(generator=generator, query="Direct query")
        trace = await Trace.from_interactions(
            Interaction(
                inputs="Trace query",
                outputs="An answer.",
                metadata={"context": "Some context."},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["query"] == "Direct query"

    async def test_custom_keys(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(
            generator=generator,
            query_key="trace.interactions[0].inputs.question",
            context_key="trace.interactions[0].metadata.chunks",
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs={"question": "Custom query"},
                outputs="An answer.",
                metadata={"chunks": ["Custom chunk."]},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["query"] == "Custom query"
        assert inputs["context"] == "Custom chunk."

    async def test_domain_context_included_in_inputs(self):
        generator = MockGenerator(passed=True, reason="Mock reason.")
        check = ContextRelevance(
            generator=generator,
            domain_context="This bot only retrieves medical documentation.",
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is aspirin?",
                outputs="A painkiller.",
                metadata={"context": "Aspirin is a painkiller."},
            ),
        )

        result = await check.run(trace)

        assert result.passed
        inputs = _assert_inputs(result)
        assert inputs["domain_context"] == (
            "This bot only retrieves medical documentation."
        )
        prompt = _rendered_prompt(generator)
        assert "<DOMAIN CONTEXT>" in prompt
        assert "This bot only retrieves medical documentation." in prompt


class TestContextRelevanceUnresolvedKeys:
    """Unresolved query/context keys return ERROR without invoking the judge."""

    async def test_empty_trace_returns_error_without_judge(self):
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = ContextRelevance(generator=generator)

        result = await check.run(Trace())

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert "query key" in (result.message or "")
        assert "trace.last.inputs" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_broken_query_key_errors_without_judge(self):
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = ContextRelevance(
            generator=generator, query_key="trace.last.metadata.does_not_exist"
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": "A language."},
            ),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert "query key" in (result.message or "")
        assert "trace.last.metadata.does_not_exist" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_missing_context_errors_without_judge(self):
        """A trace with no retrieved context errors instead of judging a placeholder."""
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(inputs="What is Python?", outputs="A language."),
        )

        result = await check.run(trace)

        assert result.status == CheckStatus.ERROR
        assert "context key" in (result.message or "")
        assert "trace.last.metadata.context" in (result.message or "")
        assert len(generator.calls) == 0

    async def test_not_does_not_invert_unresolved_key_error(self):
        generator = MockGenerator(passed=True, reason="Judge must not run.")
        check = ContextRelevance(
            generator=generator, context_key="trace.last.metadata.does_not_exist"
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": "A language."},
            ),
        )

        result = await Not(check=check).run(trace)

        assert result.status == CheckStatus.ERROR
        assert result.errored
        assert len(generator.calls) == 0
