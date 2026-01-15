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
    """Base class for all generators."""

    params: GenerationParams = Field(default_factory=GenerationParams)

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

    def chat(self, message: str, role: Role = "user") -> "ChatWorkflow":
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

    def template(self, template_name: str) -> "ChatWorkflow":
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
