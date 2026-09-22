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
    include_trace: bool = True,
) -> list[ChatMessage]:
    """Render a judge prompt with the shared-trace template flag.

    Parameters
    ----------
    prompt : str or ChatMessage or MessageTemplate or TemplateReference
        Prompt to render. Plain ``ChatMessage`` values are already rendered and
        ignore the Jinja flags.
    inputs : dict
        Template variables (trace, rule, answer, …).
    include_trace : bool, optional
        When ``False``, templates should omit the conversation ``trace`` /
        history block so the render is usable as a SOM question.

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
        "include_trace": include_trace,
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


async def _som_question_from_prompt(
    prompt: str | ChatMessage | MessageTemplate | TemplateReference,
    inputs: dict[str, Any],
) -> str:
    """Render one SOM question from a check prompt (trace omitted).

    Future batching can call this once per check while sharing the same
    :func:`_som_messages_from_trace` input across questions.
    """
    if isinstance(prompt, ChatMessage):
        return _DEFAULT_SOM_QUESTION
    question = _messages_text(await _render_prompt(prompt, inputs, include_trace=False))
    return question or _DEFAULT_SOM_QUESTION


def _som_messages_from_trace(trace: Any) -> list[ChatMessage]:
    """Fence ``trace`` as the shared SOM conversation state (batch input)."""
    text = (
        MessageTemplate(role="user", content_template="{{ trace | fence }}")
        .render(trace=trace)
        .text
        or ""
    ).strip()
    if not text:
        return []
    return [UserMessage(content=text)]


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

    Kind prefixes (``llm/`` / ``som/``) must match inference: they confirm the
    backend, they do not force a mismatched provider (e.g. ``llm/typesafe/jev``
    raises). Leaf ``parse`` methods (``LLMChatJudge.parse``, ``SOMJudge.parse``)
    only accept values that resolve to that concrete class; use
    ``BaseJudge.parse`` when the backend may be either kind.

    Examples
    --------
    Infer a System One Model from its provider prefix::

        judge = BaseJudge.parse("typesafe/jev")

    Confirm an LLM when the identifier is unambiguous::

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

        Raises
        ------
        TypeError
            If ``cls`` is a concrete subclass and ``value`` resolves to a
            different judge kind.
        """
        if isinstance(value, BaseJudge):
            judge: BaseJudge = value
        elif isinstance(value, BaseGenerator):
            judge = LLMChatJudge(generator=value)
        elif isinstance(value, BaseSOM):
            judge = SOMJudge(model=value)
        elif isinstance(value, str):
            judge = cls._from_string(value)
        elif isinstance(value, dict):
            judge = cls._from_dict(value)
        else:
            raise TypeError(
                "judge must be a BaseJudge, BaseGenerator, BaseSOM, "
                "model identifier string, JSON object, or dict"
            )
        if cls is not BaseJudge and not isinstance(judge, cls):
            raise TypeError(
                f"{cls.__name__}.parse cannot produce {type(judge).__name__}; "
                "use BaseJudge.parse when the backend kind is not fixed"
            )
        return judge

    @classmethod
    def _from_string(cls, value: str) -> "BaseJudge":
        stripped = value.strip()
        if not stripped:
            raise ValueError("judge model identifier must be a non-empty string")
        if stripped.startswith("{"):
            return cls._from_dict(json.loads(stripped))

        first, sep, rest = stripped.partition("/")
        if sep and first in BaseJudge.kinds() and rest:
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
                "Kind prefix must match the inferred backend: use 'som' with a "
                "supported SOM provider, or 'llm' with an LLM provider"
            )
        if som is not None:
            return SOMJudge(model=som)
        return LLMChatJudge(generator=Generator(model=model_id))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "BaseJudge":
        payload = dict(data)
        kind = payload.get("kind")
        if isinstance(kind, str) and kind in BaseJudge.kinds():
            return cls.model_validate(payload)
        if isinstance(kind, str) and kind in BaseSOM.kinds():
            return cls._som_judge_from_provider_dict(payload)
        if isinstance(kind, str) and kind in BaseGenerator.kinds():
            return LLMChatJudge(generator=BaseGenerator.model_validate(payload))
        if "kind" in payload:
            raise ValueError(f"Kind {kind!r} is not registered for class {cls}")

        if payload.get("generator") is not None:
            payload["kind"] = "llm"
        elif "model" in payload:
            payload["kind"] = "som"
        else:
            payload["kind"] = "llm"
        if payload.get("generator") is None:
            payload.pop("generator", None)
        return cls.model_validate(payload)

    @classmethod
    def _som_judge_from_provider_dict(cls, payload: dict[str, Any]) -> "SOMJudge":
        """Wrap a top-level SOM provider dump, peeling SOMJudge-only fields."""
        som_fields = {"pass_threshold", "timeout"}
        som_payload = {
            key: value for key, value in payload.items() if key not in som_fields
        }
        extras = {key: payload[key] for key in som_fields if key in payload}
        return SOMJudge(model=BaseSOM.model_validate(som_payload), **extras)

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


# Loose values accepted by ``BaseJudge.parse`` (and ``set_default_judge``).
type JudgeInput = BaseJudge | BaseGenerator | BaseSOM | str | dict[str, Any]
# Field annotation: static type is ``BaseJudge | None``; runtime accepts JudgeInput.
OptionalJudgeInput = Annotated[BaseJudge | None, BeforeValidator(_coerce_judge)]


@BaseJudge.register("llm")
class LLMChatJudge(BaseJudge):
    """Judge by rendering the check prompt and calling an LLM generator.

    Attributes
    ----------
    generator : BaseGenerator or None
        Generator used for evaluation. When ``None``, the global default
        generator is resolved at call time via :func:`get_default_generator`.

    Examples
    --------
    >>> from giskard.agents import Generator
    >>> from giskard.checks import LLMChatJudge
    >>> judge = LLMChatJudge(generator=Generator(model="openai/gpt-4o-mini"))
    """

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

    SOM evaluates a **question** against shared conversation **messages**.
    Bundled templates gate only the conversation block with
    ``{% if include_trace | default(true) %}`` and output schema with
    ``{% if _instr_output is defined %}``. The SOM path renders the prompt
    with ``include_trace=False`` (check variables filled in) as ``question``,
    and fences ``inputs["trace"]`` as ``messages``.

    This per-call shape is the groundwork for later batching: one trace can
    back many questions (e.g. groundedness + answer relevance) without
    re-sending the conversation. Grouping is not implemented yet.
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

        question = await _som_question_from_prompt(prompt, inputs)

        if "trace" in inputs:
            messages = _som_messages_from_trace(inputs["trace"])
            if not messages:
                return LLMCheckResult(
                    passed=False,
                    reason=(
                        "SOM judge received an empty trace after fencing. "
                        "Pass a non-empty conversation trace in inputs['trace']."
                    ),
                )
        elif isinstance(prompt, ChatMessage):
            messages = [prompt]
        else:
            return LLMCheckResult(
                passed=False,
                reason=(
                    "SOM judge requires inputs['trace'] as the shared conversation "
                    "state. Check get_inputs() should pass the Trace (batching "
                    "groundwork: one trace, many questions)."
                ),
            )

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
