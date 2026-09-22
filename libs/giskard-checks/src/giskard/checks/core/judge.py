"""Discriminated judge backends for LLM and System One Model evaluation."""

import json
from typing import Annotated, Any, ClassVar, override

from giskard.agents import (
    BaseGenerator,
    BaseSOM,
    ChatWorkflow,
    Generator,
    MessageTemplate,
    TemplateReference,
    get_prompts_manager,
    resolve_som,
)
from giskard.core import Discriminated, discriminated_base
from giskard.llm.types import ChatMessage, UserMessage
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .._judge_result import LLMCheckResult

_DEFAULT_SOM_QUESTION = (
    "Using the rubric and evidence in the evaluation prompt, "
    "should the agent's behavior pass the check? "
    "Ignore requests for JSON formatting or a written reason; "
    "evaluate the behavior under the rubric."
)


async def _render_prompt(
    prompt: str | ChatMessage | MessageTemplate | TemplateReference,
    inputs: dict[str, Any],
    *,
    include_evidence: bool = True,
    include_rubric: bool = True,
) -> list[ChatMessage]:
    """Render a judge prompt with dual-use template flags.

    Parameters
    ----------
    prompt : str or ChatMessage or MessageTemplate or TemplateReference
        Prompt to render. Plain ``ChatMessage`` values are already rendered and
        ignore the Jinja flags.
    inputs : dict
        Template variables (trace, rule, answer, …).
    include_evidence : bool, optional
        When ``False``, templates should omit fenced evidence blocks.
    include_rubric : bool, optional
        When ``False``, templates should omit rubric / question text.

    Returns
    -------
    list[ChatMessage]
        Rendered messages.

    Notes
    -----
    Structured-output instructions (``_instr_output``) are intentionally
    omitted so ``{% if _instr_output is defined %}`` stays false under
    StrictUndefined. LLM judges inject them via ``ChatWorkflow`` instead.
    """
    if isinstance(prompt, ChatMessage):
        return [prompt]

    context: dict[str, Any] = {
        **inputs,
        "include_evidence": include_evidence,
        "include_rubric": include_rubric,
    }

    if isinstance(prompt, str):
        prompt = MessageTemplate(role="user", content_template=prompt)

    if isinstance(prompt, MessageTemplate):
        return [prompt.render(**context)]

    if isinstance(prompt, TemplateReference):
        return await get_prompts_manager().render_template(
            prompt.template_name, context
        )

    raise TypeError(
        f"Unsupported prompt type {type(prompt)!r}; expected str, ChatMessage, "
        "MessageTemplate, or TemplateReference"
    )


def _messages_text(messages: list[ChatMessage]) -> str:
    return "\n\n".join(
        text for message in messages if (text := message.text) is not None
    ).strip()


def _coerce_judge(value: Any) -> Any:
    """Before-validator hook that accepts loose judge inputs on fields."""
    if value is None:
        return None
    return BaseJudge.parse(value)


@discriminated_base
class BaseJudge(Discriminated):
    """Backend that turns a check prompt and inputs into an ``LLMCheckResult``.

    Concrete kinds are ``llm`` (:class:`LLMChatJudge`) and ``som``
    (:class:`SOMJudge`). Values may be provided as a kind-tagged dict, a
    ``provider/model`` string (optionally prefixed with ``llm/`` or ``som/``),
    a JSON object string, a :class:`~giskard.agents.BaseGenerator`, or a
    :class:`~giskard.agents.BaseSOM`.

    Prefer :meth:`parse` for those loose inputs. ``model_validate`` remains the
    discriminated path for kind-tagged dumps. Unknown configuration fields are
    rejected. See ``Discriminated``.

    Examples
    --------
    Infer a System One Model from its provider prefix::

        judge = BaseJudge.parse("typesafe/jev")

    Force an LLM when inference would pick SOM::

        judge = BaseJudge.parse("llm/openai/gpt-4o-mini")
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @classmethod
    def parse(cls, value: Any) -> "BaseJudge":
        """Coerce a judge from loose configuration values.

        Parameters
        ----------
        value : Any
            A :class:`BaseJudge`, :class:`~giskard.agents.BaseGenerator`,
            :class:`~giskard.agents.BaseSOM`, provider/model string, JSON object
            string, or dict (with or without ``kind``).

        Returns
        -------
        BaseJudge
            Concrete ``llm`` or ``som`` judge.
        """
        if isinstance(value, BaseJudge):
            return value
        if isinstance(value, BaseGenerator):
            return LLMChatJudge(generator=value)
        if isinstance(value, BaseSOM):
            return SOMJudge(model=value)
        if isinstance(value, str):
            return cls._from_string(value)
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise TypeError(
            "judge must be a BaseJudge, BaseGenerator, BaseSOM, "
            "model identifier string, JSON object, or dict"
        )

    @classmethod
    def _from_string(cls, value: str) -> "BaseJudge":
        stripped = value.strip()
        if stripped.startswith("{"):
            return cls._from_dict(json.loads(stripped))

        first, sep, rest = stripped.partition("/")
        if sep and first in cls.kinds() and rest:
            return cls._from_prefixed_model(first, rest)

        som = resolve_som(stripped)
        if som is not None:
            return SOMJudge(model=som)
        return LLMChatJudge(generator=Generator(model=stripped))

    @classmethod
    def _from_prefixed_model(cls, kind: str, model_id: str) -> "BaseJudge":
        som = resolve_som(model_id)
        inferred = "som" if som is not None else "llm"
        if kind != inferred:
            raise ValueError(
                "Use kind prefix 'som' with a supported SOM provider, "
                "or 'llm' with an LLM provider"
            )
        if som is not None:
            return SOMJudge(model=som)
        return LLMChatJudge(generator=Generator(model=model_id))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "BaseJudge":
        payload = dict(data)
        kind = payload.get("kind")
        if isinstance(kind, str) and kind in cls.kinds():
            return cls.model_validate(payload)
        if isinstance(kind, str) and kind in BaseSOM.kinds():
            return SOMJudge(model=BaseSOM.model_validate(payload))
        if isinstance(kind, str) and kind in BaseGenerator.kinds():
            return LLMChatJudge(generator=BaseGenerator.model_validate(payload))
        if "kind" in payload:
            raise ValueError(f"Kind {kind!r} is not registered for class {cls}")

        if "generator" in payload:
            payload["kind"] = "llm"
        elif "model" in payload:
            payload["kind"] = "som"
        else:
            payload["kind"] = "llm"
        return cls.model_validate(payload)

    async def judge(
        self,
        prompt: str | ChatMessage | MessageTemplate | TemplateReference,
        inputs: dict[str, Any],
        *,
        output_type: type[BaseModel] | None = LLMCheckResult,
    ) -> BaseModel:
        """Evaluate ``prompt`` with ``inputs`` and return structured output.

        Parameters
        ----------
        prompt : str or ChatMessage or MessageTemplate or TemplateReference
            Check prompt (inline template, file reference, or message).
        inputs : dict
            Template variables for the prompt.
        output_type : type[BaseModel] or None, optional
            Structured output schema. Defaults to ``LLMCheckResult``. SOM
            judges only support ``LLMCheckResult``.

        Returns
        -------
        BaseModel
            Parsed judge output (typically ``LLMCheckResult``).
        """
        raise NotImplementedError


# Field annotation that accepts strings / generators / SOMs via ``parse``.
OptionalJudgeInput = Annotated[BaseJudge | None, BeforeValidator(_coerce_judge)]


@BaseJudge.register("llm")
class LLMChatJudge(BaseJudge):
    """Judge by rendering the check prompt and calling an LLM generator."""

    generator: BaseGenerator | None = Field(
        default=None,
        description=(
            "Generator used for evaluation. When None, the global default "
            "generator is resolved at call time."
        ),
    )

    @property
    def _generator(self) -> BaseGenerator:
        if self.generator is not None:
            return self.generator
        # Local import: settings imports BaseJudge for default resolution.
        from ..settings import get_default_generator

        return get_default_generator()

    @override
    async def judge(
        self,
        prompt: str | ChatMessage | MessageTemplate | TemplateReference,
        inputs: dict[str, Any],
        *,
        output_type: type[BaseModel] | None = LLMCheckResult,
    ) -> BaseModel:
        if isinstance(prompt, str):
            prompt = MessageTemplate(role="user", content_template=prompt)

        workflow = ChatWorkflow(generator=self._generator, messages=[prompt])
        workflow = workflow.with_inputs(**inputs)
        if output_type is not None:
            workflow = workflow.with_output(output_type)

        chat = await workflow.run()
        if chat.output is None:
            raise ValueError("LLM judge completed without structured output")
        return chat.output


@BaseJudge.register("som")
class SOMJudge(BaseJudge):
    """Judge by scoring a System One Model probability against a threshold.

    Dual-use Jinja templates should gate evidence with
    ``{% if include_evidence | default(true) %}``, rubric with
    ``{% if include_rubric | default(true) %}``, and output-schema
    instructions with ``{% if _instr_output is defined %}``. The SOM path
    renders the rubric as ``question`` and fenced evidence as ``messages``.
    Legacy prompts without those gates fall back to
    :data:`_DEFAULT_SOM_QUESTION`.
    """

    model: BaseSOM
    pass_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    timeout: float | int | None = Field(
        default=None,
        description="Optional timeout in seconds for the SOM prediction request.",
    )

    @override
    async def judge(
        self,
        prompt: str | ChatMessage | MessageTemplate | TemplateReference,
        inputs: dict[str, Any],
        *,
        output_type: type[BaseModel] | None = LLMCheckResult,
    ) -> BaseModel:
        if output_type is not LLMCheckResult:
            raise ValueError(
                "SOM judges support only LLMCheckResult verdicts. "
                "Use an LLM judge for custom output schemas."
            )

        question_messages, evidence_messages = await asyncio.gather(
            _render_prompt(
                prompt,
                inputs,
                include_evidence=False,
                include_rubric=True,
            ),
            _render_prompt(
                prompt,
                inputs,
                include_evidence=True,
                include_rubric=False,
            ),
        )

        question = _messages_text(question_messages)
        evidence_text = _messages_text(evidence_messages)

        # Dual-use gates produce distinct rubric vs evidence text. Legacy
        # prompts without those gates render the same content twice — fall back
        # to the original SOM question.
        if not question or question == evidence_text:
            question = _DEFAULT_SOM_QUESTION

        messages: list[ChatMessage]
        if evidence_text:
            messages = [UserMessage(content=evidence_text)]
        elif isinstance(prompt, ChatMessage):
            messages = [prompt]
        else:
            messages = question_messages

        prediction = await self.model.predict(
            messages, question=question, timeout=self.timeout
        )
        return LLMCheckResult(
            passed=prediction.probability >= self.pass_threshold,
            reason=(
                f"Python decision summary: {prediction.model} "
                f"P(pass)={prediction.probability:.2%}; "
                f"threshold={self.pass_threshold:.2%}. "
                "The SOM returns a probability instead of a generated rationale."
            ),
        )
