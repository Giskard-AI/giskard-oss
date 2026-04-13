from typing import Any, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck


@Check.register("answer_relevance")
class AnswerRelevance[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that evaluates whether the model's answer is relevant to the question.

    Uses an LLM to judge if the final answer directly and appropriately addresses the
    final question, taking into account the full conversation history. This prevents
    the judge from mis-scoring locally-reasonable answers that are off-topic given
    prior turns (e.g., a zoology answer to a programming question).

    Only the **current** turn (``question`` / ``answer``) is scored. Prior turns are
    passed as read-only history so the judge can understand user intent—not to
    penalise earlier irrelevant exchanges.

    Attributes
    ----------
    question : str | None
        The question to evaluate relevance against. When provided, takes priority
        over ``question_key``.
    question_key : JSONPathStr
        JSONPath expression to extract the question from the trace
        (default: ``"trace.last.inputs"``).
    answer : str | None
        The answer to evaluate. When provided, takes priority over ``answer_key``.
    answer_key : JSONPathStr
        JSONPath expression to extract the answer from the trace
        (default: ``"trace.last.outputs"``).
    context : str | None
        Optional domain context that describes the chatbot's purpose or scope
        (e.g., ``"This is a chatbot that answers questions about programming languages"``).
        Not extracted from the trace—must be supplied directly when needed.

    Examples
    --------
    >>> from giskard.checks import AnswerRelevance, Interaction, Scenario, Trace
    >>> scenario = (
    ...     Scenario(name="rag_relevance_multi_turn")
    ...     .interact(inputs="What is the best language?", outputs="Python")
    ...     .interact(inputs="What's Python?", outputs="A snake.")
    ...     .check(AnswerRelevance())
    ... )
    """

    question: str | None = Field(
        default=None,
        description="The question to evaluate relevance against. Takes priority over question_key.",
    )
    question_key: JSONPathStr = Field(
        default="trace.last.inputs",
        description="JSONPath to extract the question from the trace.",
    )
    answer: str | None = Field(
        default=None,
        description="The answer to evaluate. Takes priority over answer_key.",
    )
    answer_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath to extract the answer from the trace.",
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional domain context describing the chatbot's purpose or scope. "
            "Not extracted from the trace—supply directly when needed."
        ),
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(
            template_name="giskard.checks::judges/answer_relevance.j2"
        )

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict[str, Any]:
        """Build template variables from resolved inputs.

        Parameters
        ----------
        trace : Trace
            Trace for resolving inputs.

        Returns
        -------
        dict[str, str]
            Template variables with ``question``, ``answer``, ``history``, and
            ``context`` keys.
        """
        question = provided_or_resolve(
            trace,
            key=self.question_key,
            value=provide_not_none(self.question),
        )
        answer = provided_or_resolve(
            trace,
            key=self.answer_key,
            value=provide_not_none(self.answer),
        )

        return {
            "question": question,
            "answer": answer,
            "history": trace,
            "context": self.context or "",
        }
