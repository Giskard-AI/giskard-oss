"""Discriminated judge backends for LLM and System One Model evaluation."""

import json
from typing import Any, ClassVar, override

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
from giskard.agents.templates.environment import fence
from giskard.core import Discriminated, discriminated_base
from giskard.llm.types import ChatMessage, UserMessage
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from .._judge_result import LLMCheckResult


async def _render_prompt(
    prompt: str | ChatMessage | MessageTemplate | TemplateReference,
    inputs: dict[str, Any],
    *,
    include_evidence: bool = True,
    include_rubric: bool = True,
    instr_output: Any | None = None,
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
    instr_output : optional
        Structured-output instructions. Omitted when ``None`` so
        ``{% if _instr_output is defined %}`` stays false under StrictUndefined.

    Returns
    -------
    list[ChatMessage]
        Rendered messages.
    """
    if isinstance(prompt, ChatMessage):
        return [prompt]

    context: dict[str, Any] = {
        **inputs,
        "include_evidence": include_evidence,
        "include_rubric": include_rubric,
    }
    if instr_output is not None:
        context["_instr_output"] = instr_output

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


@discriminated_base
class BaseJudge(Discriminated):
    """Backend that turns a check prompt and inputs into an ``LLMCheckResult``.

    Concrete kinds are ``llm`` (:class:`LLMChatJudge`) and ``som``
    (:class:`SOMJudge`). Values may be provided as a kind-tagged dict, a
    ``provider/model`` string (optionally prefixed with ``llm/`` or ``som/``),
    a JSON object string, a :class:`~giskard.agents.BaseGenerator`, or a
    :class:`~giskard.agents.BaseSOM`.

    Unknown configuration fields are rejected. See ``Discriminated``.

    Examples
    --------
    Infer a System One Model from its provider prefix::

        judge = BaseJudge.model_validate("typesafe/jev")

    Force an LLM when inference would pick SOM::

        judge = BaseJudge.model_validate("llm/openai/gpt-4o-mini")
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        metadata = getattr(cls, "__pydantic_generic_metadata__", {})
        origin = metadata.get("origin") or cls
        # Only this base subclasses Discriminated directly; concrete kinds do not.
        if not any(base is Discriminated for base in origin.__bases__):
            return handler(source)
        return core_schema.no_info_plain_validator_function(cls._parse)

    @classmethod
    def _parse(cls, value: Any) -> "BaseJudge":
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
        if kind == "som":
            assert som is not None
            return SOMJudge(model=som)
        return LLMChatJudge(generator=Generator(model=model_id))

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "BaseJudge":
        payload = dict(data)
        kind = payload.get("kind")
        if isinstance(kind, str) and kind in cls.kinds():
            return cls._validate_kind(kind, payload)
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
        return cls._validate_kind(payload["kind"], payload)

    @classmethod
    def _validate_kind(cls, kind: str, data: dict[str, Any]) -> "BaseJudge":
        if kind == "llm":
            return LLMChatJudge.model_validate(data)
        if kind == "som":
            return SOMJudge.model_validate(data)
        raise ValueError(f"Kind {kind} is not registered for class {cls}")

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
        workflow = workflow.with_inputs(
            include_evidence=True,
            include_rubric=True,
            **inputs,
        )
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

        question_messages = await _render_prompt(
            prompt,
            inputs,
            include_evidence=False,
            include_rubric=True,
            instr_output=None,
        )
        evidence_messages = await _render_prompt(
            prompt,
            inputs,
            include_evidence=True,
            include_rubric=False,
            instr_output=None,
        )

        question = _messages_text(question_messages)
        evidence_text = _messages_text(evidence_messages)

        if not evidence_text and "trace" in inputs:
            # Custom prompts without dual-use gates: fall back to fenced trace.
            evidence_text = str(fence(inputs["trace"]))
            if not question:
                question = _messages_text(question_messages) or evidence_text

        if not question:
            question = (
                "Using the rubric and evidence, should the agent's behavior "
                "pass the check?"
            )

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
