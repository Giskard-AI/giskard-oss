import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Type

from giskard.core import Discriminated, discriminated_base
from pydantic import BaseModel, Field

from ..chat import Message, Role
from ..tools import Tool

if TYPE_CHECKING:
    from ..workflow import ChatWorkflow


class Response(BaseModel):
    message: Message
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "null"] | None
    )


class GenerationParams(BaseModel):
    """Parameters for generating a completion.

    Attributes
    ----------
    tools : list[Any], optional
        List of tools available to the model.
    """

    temperature: float = Field(default=1.0)
    max_tokens: int | None = Field(default=None)
    response_format: Type[BaseModel] | None = Field(default=None)
    tools: list[Tool] = Field(default_factory=list)


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

    # -- Completion --------------------------------------------------------

    @abstractmethod
    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response: ...

    async def complete(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
    ) -> Response:
        """Get a completion from the model.

        Parameters
        ----------
        messages : List[Message]
            List of messages to send to the model.
        params: GenerationParams | None
            Parameters for the generation.

        Returns
        -------
        Message
            The model's response message.
        """
        return await self._complete(messages, params)

    async def batch_complete(
        self, messages: list[list[Message]], params: GenerationParams | None = None
    ) -> list[Response]:
        """Get a batch of completions from the model.

        Parameters
        ----------
        messages : List[List[Message]]
            List of lists of messages to send to the model.
        params : GenerationParams, optional
            Parameters for the generation.

        Returns
        -------
        list[Response]
            A list of model's responses.
        """
        completion_requests = [self._complete(m, params) for m in messages]
        responses = await asyncio.gather(*completion_requests)
        return responses

    def chat(self, message: str, role: Role = "user") -> "ChatWorkflow[Any]":
        """Create a new chat pipeline with the given message.

        Parameters
        ----------
        message : str
            The initial message to start the chat with.

        Returns
        -------
        Pipeline
            A Pipeline object that can be used to run the completion.
        """
        from ..workflow import ChatWorkflow

        return ChatWorkflow(generator=self).chat(message, role)

    def template(self, template_name: str) -> "ChatWorkflow[Any]":
        """Create a new chat pipeline with the given message.

        Parameters
        ----------
        template_path : str
            The path to the template file.

        Returns
        -------
        Pipeline
            A Pipeline object that can be used to run the completion.
        """
        from ..workflow import ChatWorkflow

        return ChatWorkflow(generator=self).template(template_name)

    def with_params(self, **kwargs: Any) -> "BaseGenerator":
        """Create a new generator with the given parameters.

        Parameters
        ----------
        **kwargs : GenerationParamsKwargs
            The parameters to set. All fields are optional.

        Returns
        -------
        BaseGenerator
            A new generator with the given parameters.
        """
        generator = self.model_copy(deep=True)
        generator.params = generator.params.model_copy(update=kwargs)
        return generator
