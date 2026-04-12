"""Unit tests for the ContextRelevance check.

Tests cover:
- Relevant context passes; irrelevant context fails
- List-of-strings context handled correctly
- Multi-turn: pronoun / reference resolution via history
- Direct query / context values take priority over trace extraction
- Custom JSONPath keys
- Domain context forwarded to template inputs
- Empty trace handled gracefully
"""

import json
from typing import Any, override

from giskard.agents.chat import Message
from giskard.agents.generators._types import Response
from giskard.agents.generators.base import BaseGenerator, GenerationParams
from giskard.checks import CheckStatus, ContextRelevance, Interaction, Trace
from pydantic import Field


class MockGenerator(BaseGenerator):
    passed: bool
    reason: str | None = None
    calls: list[list[Message]] = Field(default_factory=list)

    @override
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        self.calls.append(messages)
        return Response(
            message=Message(
                role="assistant",
                content=json.dumps({"passed": self.passed, "reason": self.reason}),
            ),
            finish_reason="stop",
        )


class TestContextRelevanceBasic:
    """Standard RAG pass / fail behaviour."""

    async def test_relevant_context_passes(self):
        """Context containing the answer to the query should pass."""
        generator = MockGenerator(
            passed=True,
            reason="Context contains information about Python installation.",
        )
        check = ContextRelevance(
            generator=generator,
            query="How do I install Python?",
            retrieved_context="To install Python, download the installer from python.org.",
        )
        result = await check.run(Trace())

        assert result.status == CheckStatus.PASS
        assert result.passed
        assert (
            result.details["reason"]
            == "Context contains information about Python installation."
        )

    async def test_irrelevant_context_fails(self):
        """Context unrelated to the query should fail."""
        generator = MockGenerator(
            passed=False,
            reason="Context is about cooking, not Python installation.",
        )
        check = ContextRelevance(
            generator=generator,
            query="How do I install Python?",
            retrieved_context="Preheat oven to 180°C and bake for 25 minutes.",
        )
        result = await check.run(Trace())

        assert result.status == CheckStatus.FAIL
        assert result.failed

    async def test_llm_called_once(self):
        """Exactly one LLM call should be made per check run."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query="What is Flask?",
            retrieved_context="Flask is a lightweight Python web framework.",
        )
        await check.run(Trace())

        assert len(generator.calls) == 1


class TestContextRelevanceListHandling:
    """Context can be a list of strings or a single string."""

    async def test_list_context_passed_directly(self):
        """A list of context strings supplied directly is handled."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query="What is Python?",
            retrieved_context=[
                "Python is a high-level programming language.",
                "Python was created by Guido van Rossum.",
            ],
        )
        result = await check.run(Trace())

        assert result.passed
        assert "Python is a high-level" in result.details["inputs"]["context"]

    async def test_list_context_from_trace(self):
        """A list stored in trace metadata is extracted and stringified."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="Python is a language.",
                metadata={
                    "context": [
                        "Python is a high-level language.",
                        "It supports multiple paradigms.",
                    ]
                },
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert "Python is a high-level" in result.details["inputs"]["context"]

    async def test_single_string_context_from_trace(self):
        """A single string stored in trace metadata is accepted."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="Python is a language.",
                metadata={"context": "Python is a high-level programming language."},
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert "Python is a high-level" in result.details["inputs"]["context"]


class TestContextRelevanceMultiTurn:
    """Multi-turn pronoun / reference resolution."""

    async def test_pronoun_resolution_relevant(self):
        """'How do I install it?' resolves via history; relevant context passes."""
        generator = MockGenerator(
            passed=True,
            reason="Context contains installation instructions for Giskard checks.",
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Giskard checks?",
                outputs="Giskard checks is a Python library for LLM evaluation.",
                metadata={
                    "context": ["Giskard checks is a Python evaluation library."]
                },
            ),
            Interaction(
                inputs="How do I install it?",
                outputs="Use pip install giskard-checks.",
                metadata={"context": ["pip install giskard-checks"]},
            ),
        )
        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed

    async def test_history_included_in_inputs(self):
        """Prior turn must appear in the history template variable."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Giskard?",
                outputs="An LLM evaluation library.",
                metadata={"context": ["Giskard is an LLM evaluation tool."]},
            ),
            Interaction(
                inputs="How do I install it?",
                outputs="pip install giskard-checks",
                metadata={"context": ["pip install giskard-checks"]},
            ),
        )
        result = await check.run(trace)

        assert result.passed
        history = result.details["inputs"]["history"]
        assert "What is Giskard?" in history
        assert "An LLM evaluation library." in history
        # Current turn should NOT appear in history
        assert "How do I install it?" not in history

    async def test_single_turn_history_is_empty(self):
        """First interaction has no prior history."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is Python?",
                outputs="A language.",
                metadata={"context": ["Python is a programming language."]},
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert result.details["inputs"]["history"] == ""

    async def test_prior_irrelevant_turn_does_not_penalise_current(self):
        """A prior irrelevant exchange must not cause a relevant context to fail."""
        generator = MockGenerator(
            passed=True,
            reason="Context is relevant to the current query.",
        )
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is the best language?",
                outputs="You should cook lasagna.",
                metadata={"context": ["Lasagna recipe steps."]},
            ),
            Interaction(
                inputs="Is Python a language or an animal?",
                outputs="Both.",
                metadata={
                    "context": [
                        "Python is a programming language.",
                        "Python is also a type of snake.",
                    ]
                },
            ),
        )
        result = await check.run(trace)

        assert result.status == CheckStatus.PASS
        assert result.passed


class TestContextRelevanceInputResolution:
    """Direct values vs. trace extraction."""

    async def test_direct_query_and_context_used(self):
        """Directly supplied query/context take priority over trace."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query="Direct query",
            retrieved_context="Direct context",
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs="Trace query",
                outputs="Trace answer",
                metadata={"context": "Trace context"},
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert result.details["inputs"]["query"] == "Direct query"
        assert result.details["inputs"]["context"] == "Direct context"

    async def test_query_and_context_extracted_from_trace(self):
        """When no direct values given, query/context extracted from trace."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)
        trace = await Trace.from_interactions(
            Interaction(
                inputs="What is AI?",
                outputs="AI is artificial intelligence.",
                metadata={"context": "AI stands for artificial intelligence."},
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert result.details["inputs"]["query"] == "What is AI?"
        assert "AI stands for" in result.details["inputs"]["context"]

    async def test_custom_keys(self):
        """Custom JSONPath keys should resolve correctly."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query_key="trace.interactions[0].inputs.question",
            context_key="trace.interactions[0].metadata.docs",
        )
        trace = await Trace.from_interactions(
            Interaction(
                inputs={"question": "Custom query"},
                outputs="Custom answer",
                metadata={"docs": ["Custom context doc"]},
            )
        )
        result = await check.run(trace)

        assert result.passed
        assert result.details["inputs"]["query"] == "Custom query"
        assert "Custom context doc" in result.details["inputs"]["context"]

    async def test_empty_trace_no_crash(self):
        """Empty trace should not raise — NoMatch values are stringified."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(generator=generator)

        result = await check.run(Trace())

        assert result.passed
        assert "No match for key" in result.details["inputs"]["query"]
        assert "No match for key" in result.details["inputs"]["context"]


class TestContextRelevanceDomainContext:
    """Optional domain context is forwarded to template inputs."""

    async def test_domain_context_included_in_inputs(self):
        """Supplied domain context must appear in template inputs."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query="What is Flask?",
            retrieved_context="Flask is a Python web framework.",
            domain_context="This bot only retrieves Python programming documentation.",
        )
        result = await check.run(Trace())

        assert result.passed
        assert result.details["inputs"]["domain_context"] == (
            "This bot only retrieves Python programming documentation."
        )

    async def test_no_domain_context_is_empty_string(self):
        """When no domain context supplied, template input is empty string."""
        generator = MockGenerator(passed=True, reason=None)
        check = ContextRelevance(
            generator=generator,
            query="What is Flask?",
            retrieved_context="Flask is a web framework.",
        )
        result = await check.run(Trace())

        assert result.passed
        assert result.details["inputs"]["domain_context"] == ""
