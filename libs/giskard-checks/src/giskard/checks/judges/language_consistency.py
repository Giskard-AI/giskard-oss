from typing import Any, override

from giskard.agents import TemplateReference
from pydantic import Field
from pydantic.experimental.missing_sentinel import MISSING

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, NoMatch, provided_or_resolve
from ..core.result import CheckResult
from ._inputs import ResolvableInput, error_if_unresolved
from .base import BaseLLMCheck


@Check.register("language_consistency")
class LanguageConsistency[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that flags a language mismatch between user and agent.

    Uses an LLM judge to compare the language of the agent's response against a
    reference language: an explicit ``expected_language`` when set, otherwise the
    language of the most recent user message that carries a usable language
    signal. The check fails only when a substantial part of the response's
    free-flowing prose is in the wrong language.

    Isolated technical terms, proper nouns, code blocks, and verbatim quotations
    are never held against the language judgement. A response with no language
    signal (a bare number, a URL, "OK") passes by default. Jargon borrowings that
    keep a sentence's matrix language intact ("j'ai run le check") are not a
    switch; whole clauses in another language are.

    Attributes
    ----------
    expected_language : str | MISSING
        Explicit reference language (e.g. ``"French"``, ``"fr"``). When set, it
        overrides the user's language and ``user_input_key`` is not required to
        resolve. Not extracted from the trace—supply directly when needed.
    user_input : str | MISSING
        The user input whose language sets the reference. Takes priority over
        ``user_input_key``. If omitted, extracted from the trace using
        ``user_input_key``.
    user_input_key : JSONPathStr
        JSONPath expression to extract the user input from the trace
        (default: ``"trace.last.inputs"``). The judge still walks the history
        backwards to find the last turn carrying a language signal, so this may
        legitimately resolve to a signal-less message.
    output : str | MISSING
        The agent response to evaluate. Takes priority over ``target_key``. If
        omitted, extracted from the trace using ``target_key``.
    target_key : JSONPathStr
        JSONPath expression to extract the response from the trace
        (default: ``"trace.last.outputs"``).
    generator : BaseGenerator | None
        Generator for LLM evaluation (inherited from BaseLLMCheck).

    Examples
    --------
    Detect a language mismatch from a trace:

    >>> from giskard.checks import LanguageConsistency, Scenario
    >>> scenario = (
    ...     Scenario(name="language_check")
    ...     .interact(
    ...         inputs="Quelle est la capitale de la France ?",
    ...         outputs="Paris is the capital of France.",
    ...     )
    ...     .check(LanguageConsistency())
    ... )

    Pin an expected language regardless of the user's:

    >>> from giskard.agents import Generator
    >>> check = LanguageConsistency(
    ...     output="Voici votre réponse.",
    ...     expected_language="English",
    ...     generator=Generator(model="openai/gpt-4o"),
    ... )
    """

    expected_language: str | MISSING = Field(
        default=MISSING,
        description=(
            "Explicit reference language that overrides the user's language. "
            "Not extracted from the trace—supply directly when needed."
        ),
    )
    user_input: str | MISSING = Field(
        default=MISSING,
        description=(
            "The user input whose language sets the reference. "
            "Takes priority over input_key."
        ),
    )
    user_input_key: JSONPathStr = Field(
        default="trace.last.inputs",
        description="JSONPath to extract the user input from the trace.",
    )
    output: str | MISSING = Field(
        default=MISSING,
        description="The agent response to evaluate. Takes priority over target_key.",
    )
    target_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath to extract the agent response from the trace.",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        """Return the bundled prompt template for language-consistency evaluation."""
        return TemplateReference(
            template_name="giskard.checks::judges/language_consistency.j2"
        )

    @override
    async def run(self, trace: TraceType) -> CheckResult:
        """Return ERROR when a required key does not resolve; else run the judge.

        Guarding here—before ``super().run()``—means a misconfigured key costs no
        judge call, and ERROR (rather than FAIL) keeps ``Not(...)`` from
        laundering a broken key into a green result.

        ``target_key`` is always required. ``user_input_key`` is required only
        when no ``expected_language`` is set: with an explicit reference language
        the user input is never consulted. The guard only checks that the key
        resolves, not that the message carries a language signal—the judge walks
        the history backwards for that.
        """
        inputs = [ResolvableInput("output", self.target_key, self.output, "target_key")]
        if self.expected_language is MISSING:
            inputs.append(
                ResolvableInput("user_input", self.user_input_key, self.user_input)
            )
        if early := error_if_unresolved(trace, *inputs):
            return early
        return await super().run(trace)

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables for the language-consistency judge prompt.

        Returns
        -------
        dict[str, Any]
            Template variables with ``trace``, ``history``, ``user_input``,
            ``output``, and ``expected_language`` keys. ``history`` is always
            passed (unlike some judges) because the reference language may come
            from a turn earlier than the last one. When ``expected_language`` is
            set, an unresolved ``user_input`` is normalised to an empty string so
            the template drops the ``<USER INPUT>`` section instead of rendering a
            ``NoMatch`` marker.
        """
        resolved_input = provided_or_resolve(
            trace,
            key=self.user_input_key,
            value=self.user_input,
        )
        return {
            "trace": trace,
            "history": trace,
            "user_input": "" if isinstance(resolved_input, NoMatch) else resolved_input,
            "output": provided_or_resolve(
                trace,
                key=self.target_key,
                value=self.output,
            ),
            "expected_language": (
                self.expected_language if self.expected_language is not MISSING else ""
            ),
        }
