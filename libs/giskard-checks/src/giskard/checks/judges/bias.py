from typing import Literal, override

from giskard.agents.workflow import TemplateReference
from giskard.core import provide_not_none
from pydantic import Field

from ..core import Trace
from ..core.check import Check
from ..core.extraction import JSONPathStr, provided_or_resolve
from .base import BaseLLMCheck

ProtectedAttribute = Literal[
    "gender",
    "race",
    "age",
    "religion",
    "nationality",
    "disability",
    "sexual_orientation",
]

DEFAULT_PROTECTED_ATTRIBUTES: tuple[ProtectedAttribute, ...] = (
    "gender",
    "race",
    "age",
    "religion",
    "nationality",
    "disability",
    "sexual_orientation",
)


@Check.register("bias")
class Bias[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    BaseLLMCheck[InputType, OutputType, TraceType]
):
    """LLM-based check that detects biased content in model outputs.

    Uses an LLM judge to determine whether the model output contains bias
    including stereotyping, discrimination, or unfair representation across
    configurable demographic dimensions.

    Attributes
    ----------
    output : str | None
        The text to evaluate for bias. If None, extracted from the trace
        using ``output_key``.
    output_key : JSONPathStr
        JSONPath expression to extract the output from the trace
        (default: ``"trace.last.outputs"``).
    protected_attributes : list[ProtectedAttribute]
        Demographic dimensions to evaluate for bias. Defaults to all built-in
        attributes: ``gender``, ``race``, ``age``, ``religion``, ``nationality``,
        ``disability``, ``sexual_orientation``.
    context_key : JSONPathStr | None
        Optional JSONPath expression to extract input/context for evaluating
        relative bias. When provided, the context is forwarded to the judge.

    Examples
    --------
    Check for all bias dimensions using a trace:

    >>> from giskard.checks import Bias, Scenario
    >>> scenario = (
    ...     Scenario(name="bias_check")
    ...     .interact(inputs="Describe a software engineer", outputs="He codes all day.")
    ...     .check(Bias())
    ... )

    Check only for gender and race bias:

    >>> from giskard.agents.generators import Generator
    >>> check = Bias(
    ...     output="She is naturally better at caring roles.",
    ...     protected_attributes=["gender", "race"],
    ...     generator=Generator(model="openai/gpt-4o"),
    ... )
    """

    output: str | None = Field(
        default=None,
        description="The text to evaluate for bias. If None, extracted from the trace using output_key.",
    )
    output_key: JSONPathStr = Field(
        default="trace.last.outputs",
        description="JSONPath expression to extract the output from the trace.",
    )
    protected_attributes: list[ProtectedAttribute] = Field(
        default_factory=lambda: list(DEFAULT_PROTECTED_ATTRIBUTES),
        description=(
            "Demographic dimensions to evaluate for bias. "
            "Defaults to all built-in attributes: "
            "gender, race, age, religion, nationality, disability, sexual_orientation."
        ),
    )
    context_key: JSONPathStr | None = Field(
        default=None,
        description="Optional JSONPath to extract input/context for evaluating relative bias.",
    )

    @override
    def get_prompt(self) -> TemplateReference:
        return TemplateReference(template_name="giskard.checks::judges/bias.j2")

    @override
    async def get_inputs(self, trace: Trace[InputType, OutputType]) -> dict:
        resolved_output = str(
            provided_or_resolve(
                trace,
                key=self.output_key,
                value=provide_not_none(self.output),
            )
        )

        context = ""
        if self.context_key is not None:
            resolved_context = provided_or_resolve(trace, key=self.context_key)
            context = str(resolved_context)

        return {
            "output": resolved_output,
            "protected_attributes": self.protected_attributes,
            "context": context,
        }
