import asyncio
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Any, Self

from giskard.core import Discriminated, discriminated_base
from pydantic import Field

from ..chat import Message, Role
from ..tools import Tool
from ._types import FinishReason, GenerationParams, Response
from .middleware import CompletionMiddleware, NextFn

if TYPE_CHECKING:
    from ..workflow import ChatWorkflow


@discriminated_base
class BaseGenerator(Discriminated, ABC):
    """Base class for all generators.

    Generators act as **protocol adapters**: they own all translation between
    the internal ``Message`` format and whatever wire format the LLM provider
    expects.  The three translation methods—``serialize_tools``,
    ``serialize_messages``, and ``deserialize_response``—are the extension
    points that provider-specific subclasses override.  Workflow, tool, and
    chat code must never call provider APIs directly; they work exclusively
    with ``Message`` objects and delegate wire translation to the generator.
    """

    params: GenerationParams = Field(default_factory=GenerationParams)
    middleware: list[CompletionMiddleware] = Field(default_factory=list)

    # -- Protocol adapter methods ------------------------------------------

    def serialize_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert internal ``Tool`` objects to the provider's wire format.

        Override in subclasses to produce a different tool schema layout
        (e.g. Anthropic's tool format).

        Parameters
        ----------
        tools : list[Tool]
            Tools to serialize.

        Returns
        -------
        list[dict[str, Any]]
            Tool definitions in the provider's expected format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in tools
        ]

    def serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal ``Message`` objects to the provider's wire format.

        Override in subclasses to reshape messages for providers that use a
        different message layout (e.g. Anthropic batches tool results into a
        single ``role="user"`` message with ``tool_result`` content blocks).

        Parameters
        ----------
        messages : list[Message]
            Messages to serialize.

        Returns
        -------
        list[dict[str, Any]]
            Messages in the provider's expected format.
        """
        return [
            m.model_dump(include={"role", "content", "tool_calls", "tool_call_id"})
            for m in messages
        ]

    def deserialize_response(self, raw: Any) -> Message:
        """Convert a provider's raw response into an internal ``Message``.

        Override in subclasses to handle provider-specific response shapes.

        Parameters
        ----------
        raw : Any
            The raw response object from the provider.

        Returns
        -------
        Message
            An internal Message instance.
        """
        data = raw if isinstance(raw, dict) else raw.model_dump()
        return Message.model_validate(data)

    # -- Completion pipeline -----------------------------------------------

    def _resolve_params(
        self, params: GenerationParams | None
    ) -> tuple[dict[str, Any], list[Tool]]:
        """Merge ``self.params`` with per-call *params* overrides.

        Returns
        -------
        tuple[dict[str, Any], list[Tool]]
            A ``(wire_params, tools)`` pair ready for serialization.
        """
        merged = self.params.model_dump(exclude={"tools"})
        if params is not None:
            merged.update(params.model_dump(exclude={"tools"}, exclude_unset=True))
        tools = self.params.tools + (params.tools if params is not None else [])
        return merged, tools

    @abstractmethod
    async def _call_model(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> tuple[Any, FinishReason]:
        """Call the provider and return the raw response.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Messages in the provider's wire format (from ``serialize_messages``).
        tools : list[dict[str, Any]]
            Tool definitions in the provider's wire format (from ``serialize_tools``).
            Empty list when no tools are available.
        params : dict[str, Any]
            Merged generation parameters (temperature, max_tokens, etc.).

        Returns
        -------
        tuple[Any, FinishReason]
            ``(raw_message, finish_reason)`` — the raw message will be passed
            to ``deserialize_response``.
        """
        raise NotImplementedError

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        wire_params, tools = self._resolve_params(params)
        wire_tools = self.serialize_tools(tools) if tools else []
        wire_messages = self.serialize_messages(messages)
        raw_message, finish_reason = await self._call_model(
            wire_messages, wire_tools, wire_params
        )
        return Response(
            message=self.deserialize_response(raw_message),
            finish_reason=finish_reason,
        )

    async def complete(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
    ) -> Response:
        """Get a completion from the model.

        Parameters
        ----------
        messages : list[Message]
            List of messages to send to the model.
        params: GenerationParams | None
            Parameters for the generation.

        Returns
        -------
        Response
            The model's response.
        """
        chain = self._build_chain(self._complete)
        return await chain(messages, params)

    def _build_chain(self, core: NextFn) -> NextFn:
        """Fold ``self.middleware`` around *core*, first element = outermost."""

        def _wrap(next_fn: NextFn, mw: CompletionMiddleware) -> NextFn:
            async def _wrapped(
                messages: list[Message], params: GenerationParams | None
            ) -> Response:
                return await mw.call(messages, params, next_fn)

            return _wrapped

        return reduce(_wrap, reversed(self.middleware), core)

    async def batch_complete(
        self, messages: list[list[Message]], params: GenerationParams | None = None
    ) -> list[Response]:
        """Get a batch of completions from the model.

        Parameters
        ----------
        messages : list[list[Message]]
            List of lists of messages to send to the model.
        params : GenerationParams | None, optional
            Parameters for the generation.

        Returns
        -------
        list[Response]
            A list of model's responses.
        """
        completion_requests = [self.complete(m, params) for m in messages]
        responses = await asyncio.gather(*completion_requests)
        return responses

    def chat(self, message: str, role: Role = "user") -> "ChatWorkflow[Any]":
        """Create a new chat workflow with the given message.

        Parameters
        ----------
        message : str
            The initial message to start the chat with.

        Returns
        -------
        ChatWorkflow
            A ChatWorkflow that can be used to run the completion.
        """
        from ..workflow import ChatWorkflow

        return ChatWorkflow(generator=self).chat(message, role)

    def template(self, template_name: str) -> "ChatWorkflow[Any]":
        """Create a new chat workflow from a template.

        Parameters
        ----------
        template_name : str
            The name of the template.

        Returns
        -------
        ChatWorkflow
            A ChatWorkflow that can be used to run the completion.
        """
        from ..workflow import ChatWorkflow

        return ChatWorkflow(generator=self).template(template_name)

    def with_params(self, **kwargs: Any) -> Self:
        """Create a new generator with the given parameters.

        Parameters
        ----------
        **kwargs
            The parameters to set. All fields are optional.

        Returns
        -------
        Self
            A new generator with the given parameters.
        """
        generator = self.model_copy(deep=True)
        generator.params = generator.params.model_copy(update=kwargs)
        return generator
