from typing import Any, override

from giskard.agents import TemplateReference
from pydantic import Field
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve, resolve
from ..core.result import CheckResult
from ._inputs import ResolvableInput, error_if_unresolved
from .base import BaseLLMCheck, format_prompt_text


@Check.register("context_relevance")
class ContextRelevance[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates whether the retrieved context fits the query.

    Uses an LLM to judge if the context retrieved for the current turn contains
    the information needed to answer the current query, taking the conversation
    history into account so that underspecified queries can be disambiguated
    ("How do I install it?" resolves "it" against earlier turns).

    Only the **current** exchange (``query`` / retrieved context) is scored. Prior
    turns are passed as read-only history so the judge can resolve references—not
    to penalise earlier irrelevant exchanges.

    Attributes
    ----------
    query : str | None
        The query to evaluate the retrieved context against. When provided, takes
        priority over ``query_key``. If omitted, extracted from the trace using
        ``query_key``.
    query_key : JSONPathStr
        JSONPath expression to extract the query from the trace
        (default: ``"trace.last.inputs"``).
    context_key : JSONPathStr
        JSONPath expression to extract the retrieved context from the trace
        (default: ``"trace.last.metadata.context"``). The resolved value may be a
        single string or a list of strings.
    history : JSONPathStr
        JSONPath expression to extract the prior turns used for disambiguation
        (default: ``"trace.interactions[:-1]"``). Resolving to nothing omits the
        ``<CONVERSATION HISTORY>`` section from the prompt entirely.
    domain_context : str | None
        Optional domain context describing the assistant's purpose or scope
        (e.g., ``"This bot only retrieves medical documentation"``). Not extracted
        from the trace—supply directly when needed.

    Notes
    -----
    When ``query_key`` or ``context_key`` matches nothing in the trace, the check
    returns ``CheckStatus.ERROR`` without invoking the judge. History is supporting
    context only: the judge is instructed to score the resolved query/context and
    never to substitute a query inferred from history, so a misconfigured key
    surfaces as an error instead of being silently rescued.

    Examples
    --------
    >>> from giskard.checks import ContextRelevance, Scenario
    >>> scenario = (
    ...     Scenario(name="retrieval_quality")
    ...     .interact(
    ...         inputs="What is Python?",
    ...         outputs="Python is a language.",
    ...         metadata={"context": ["Python is high-level..."]},
    ...     )
    ...     .interact(
    ...         inputs="How do I install it?",
    ...         metadata={"context": ["To install Python, use pyenv..."]},
    ...     )
    ...     .check(ContextRelevance())
    ... )
    """

    query: str | None = Field(
        default=None,
        description="The query to evaluate the context against. Takes priority over query_key.",
    )
    query_key: JSONPathStr = Field(
        default="trace.last.inputs",
        description="JSONPath to extract the query from the trace.",
    )
    context_key: JSONPathStr = Field(
        default="trace.last.metadata.context",
        description="JSONPath to extract the retrieved context from the trace.",
    )
    history: JSONPathStr = Field(
        default="trace.interactions[:-1]",
        description="JSONPath to extract the prior turns used for disambiguation.",
    )
    domain_context: str | None = Field(
        default=None,
        description=(
            "Optional domain context describing the assistant's purpose or scope. "
            "Not extracted from the trace—supply directly when needed."
        ),
    )

    @property
    def _query_value(self) -> Any:
        """Return the directly-provided query, or MISSING so the key is used."""
        return MISSING if self.query is None else self.query

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/context_relevance.j2"
        )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Return ERROR when query/context keys do not resolve; else run the judge.

        Guarding here—before ``super().run()``—means a misconfigured key costs no
        judge call. The query is checked first so a fully missing trace reports
        the key that best explains the misconfiguration.
        """
        if early := error_if_unresolved(
            trace,
            ResolvableInput("query", self.query_key, self._query_value),
            ResolvableInput("context", self.context_key, MISSING),
        ):
            return early
        return await super().run(trace)

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, Any]
            Template variables with ``query``, ``context``, ``history`` and
            ``domain_context`` keys. Context resolving to a list of strings is
            joined into a single newline-separated block.
        """
        history = resolve(trace, self.history)
        # A history key that matches nothing must not leak a NoMatch placeholder
        # into the prompt; render it as empty so the template drops the section.
        if isinstance(history, NoMatch):
            history = ""

        return {
            "query": format_prompt_text(
                provided_or_resolve(
                    trace,
                    key=self.query_key,
                    value=self._query_value,
                )
            ),
            "context": format_prompt_text(resolve(trace, self.context_key)),
            "history": format_prompt_text(history) if history else "",
            "domain_context": self.domain_context or "",
        }
