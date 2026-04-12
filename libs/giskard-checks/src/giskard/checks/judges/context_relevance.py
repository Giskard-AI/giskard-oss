from typing import override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("context_relevance")
class ContextRelevance[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates whether retrieved context is relevant to the query.

    Uses an LLM to judge if the retrieved context contains the information needed to
    answer the user's query, taking into account full conversation history for
    pronoun and reference resolution in multi-turn dialogues.

    Only the **current** turn's query and context are scored. Prior turns are passed
    as read-only history to disambiguate underspecified queries (e.g. "How do I
    install it?" resolves to the topic established in earlier turns).

    Attributes
    ----------
    query : str | None
        The user query to evaluate context relevance against. When provided, takes
        priority over ``query_key``.
    query_key : JSONPathStr
        JSONPath expression to extract the query from the trace
        (default: ``"trace.last.inputs"``).
    retrieved_context : str | list[str] | None
        The retrieved context to evaluate. When provided, takes priority over
        ``context_key``. Accepts a single string or a list of strings.
    context_key : JSONPathStr
        JSONPath expression to extract the retrieved context from the trace
        (default: ``"trace.last.metadata.context"``).
    domain_context : str | None
        Optional high-level description of the system's intended domain or purpose
        (e.g., ``"This bot only retrieves medical documentation"``). Not extracted
        from the trace — supply directly when needed.

    Examples
    --------
    >>> from giskard.checks import ContextRelevance, Interaction, Scenario
    >>> scenario = (
    ...     Scenario(name="retrieval_quality")
    ...     .interact(
    ...         inputs="What is Python?",
    ...         outputs="Python is a language.",
    ...         metadata={"context": ["Python is a high-level programming language."]}
    ...     )
    ...     .interact(
    ...         inputs="How do I install it?",
    ...         outputs="Use pip install.",
    ...         metadata={"context": ["To install Python, use pyenv or the official installer."]}
    ...     )
    ...     .check(ContextRelevance())
    ... )
    """

    query: str | None = Field(
        default=None,
        description="The query to evaluate context relevance against. Takes priority over query_key.",
    )
    query_key: JSONPathStr = Field(
        default="trace.last.inputs",
        description="JSONPath to extract the query from the trace.",
    )
    retrieved_context: str | list[str] | None = Field(
        default=None,
        description=(
            "The retrieved context to evaluate. Takes priority over context_key. "
            "Accepts a single string or a list of strings."
        ),
    )
    context_key: JSONPathStr = Field(
        default="trace.last.metadata.context",
        description="JSONPath to extract the retrieved context from the trace.",
    )
    domain_context: str | None = Field(
        default=None,
        description=(
            "Optional high-level description of the system's intended domain or purpose. "
            "Not extracted from the trace — supply directly when needed."
        ),
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/context_relevance.j2"
        )

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, str]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables with ``query``, ``context``, ``history``, and
            ``domain_context`` keys.
        """
        query = str(
            provided_or_resolve(
                trace,
                key=self.query_key,
                value=provide_not_none(self.query),
            )
        )
        context = str(
            provided_or_resolve(
                trace,
                key=self.context_key,
                value=provide_not_none(self.retrieved_context),
            )
        )

        # Build plain-text history of prior turns for query disambiguation.
        history_turns = trace.interactions[:-1] if trace.interactions else []
        if history_turns:
            history_lines: list[str] = []
            for i, interaction in enumerate(history_turns, start=1):
                history_lines.append(
                    f"Turn {i}:\n  User: {interaction.inputs}\n  Assistant: {interaction.outputs}"
                )
            history = "\n\n".join(history_lines)
        else:
            history = ""

        return {
            "query": query,
            "context": context,
            "history": history,
            "domain_context": self.domain_context or "",
        }
