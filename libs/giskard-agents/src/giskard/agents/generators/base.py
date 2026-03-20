import asyncio
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Any, Self

from giskard.core import BaseRateLimiter, Discriminated, discriminated_base
from pydantic import Field

from ..chat import Message, Role
from ._types import GenerationParams, Response
from .middleware import (
    CompletionMiddleware,
    NextFn,
    RateLimiterMiddleware,
    RetryMiddleware,
    RetryPolicy,
)

if TYPE_CHECKING:
    from ..workflow import ChatWorkflow


@discriminated_base
class BaseGenerator(Discriminated, ABC):
    """Base class for all generators.

    Each subclass is responsible for translating between the internal
    ``Message`` / ``Tool`` objects and whatever wire format its provider
    expects.  Workflow, tool, and chat code work exclusively with
    ``Message`` objects and never call provider APIs directly.
    """

    params: GenerationParams = Field(default_factory=GenerationParams)
    retry_policy: RetryPolicy | None = Field(default=None)
    rate_limiter: BaseRateLimiter | None = Field(default=None)
    middlewares: list[CompletionMiddleware] = Field(default_factory=list)

    # -- Completion pipeline -----------------------------------------------

    @abstractmethod
    async def _call_model(
        self,
        messages: list[Message],
        params: GenerationParams,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        """Call the provider and return a Response.

        Subclasses handle all serialization/deserialization internally.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages in internal format.
        params : GenerationParams
            Merged generation parameters (including tools).
        metadata : dict[str, Any] | None
            Optional metadata from the caller (e.g. trace IDs, tags).

        Returns
        -------
        Response
            The model's response including message, finish reason, and metadata.
        """
        raise NotImplementedError

    async def _complete(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        """Template method: merge generation params and delegate to the subclass.

        This is the core of the middleware pipeline.  Do not override in
        production subclasses — override ``_call_model`` instead.
        """
        merged = self.params.merge(params)
        return await self._call_model(messages, merged, metadata)

    async def complete(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        """Get a completion from the model.

        Parameters
        ----------
        messages : list[Message]
            List of messages to send to the model.
        params: GenerationParams | None
            Parameters for the generation.
        metadata : dict[str, Any] | None
            Optional metadata to pass through the completion pipeline.

        Returns
        -------
        Response
            The model's response.
        """
        chain = self._build_chain(self._complete)
        return await chain(messages, params, metadata)

    def _build_chain(self, core: NextFn) -> NextFn:
        """Compose built-in retry/rate-limiter and custom middlewares around *core*."""
        built_in: list[CompletionMiddleware] = []
        if retry_mw := self._create_retry_middleware():
            built_in.append(retry_mw)
        if self.rate_limiter is not None:
            built_in.append(RateLimiterMiddleware(rate_limiter=self.rate_limiter))

        all_mw = built_in + list(self.middlewares)

        def _wrap(next_fn: NextFn, mw: CompletionMiddleware) -> NextFn:
            async def _wrapped(
                messages: list[Message],
                params: GenerationParams | None,
                metadata: dict[str, Any] | None,
            ) -> Response:
                return await mw.call(messages, params, metadata, next_fn)

            return _wrapped

        return reduce(_wrap, reversed(all_mw), core)

    def _create_retry_middleware(self) -> RetryMiddleware | None:
        """Create the retry middleware from ``retry_policy``.

        Subclasses can override to return a provider-specific middleware.
        """
        if self.retry_policy is None:
            return None
        return RetryMiddleware(retry_policy=self.retry_policy)

    def with_retries(
        self,
        max_attempts: int,
        *,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ) -> Self:
        """Return a copy with an updated retry policy."""
        updates: dict[str, Any] = {"max_attempts": max_attempts}
        if base_delay is not None:
            updates["base_delay"] = base_delay
        if max_delay is not None:
            updates["max_delay"] = max_delay

        policy = (self.retry_policy or RetryPolicy()).model_copy(update=updates)
        return self.model_copy(update={"retry_policy": policy})

    def with_rate_limiter(self, rate_limiter: BaseRateLimiter | str | None) -> Self:
        """Return a copy with the given rate limiter."""
        if isinstance(rate_limiter, str):
            rate_limiter = BaseRateLimiter.from_id(rate_limiter)
        return self.model_copy(update={"rate_limiter": rate_limiter})

    async def batch_complete(
        self,
        messages: list[list[Message]],
        params: GenerationParams | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Response]:
        """Get a batch of completions from the model.

        Parameters
        ----------
        messages : list[list[Message]]
            List of lists of messages to send to the model.
        params : GenerationParams | None, optional
            Parameters for the generation.
        metadata : dict[str, Any] | None, optional
            Optional metadata to pass through the completion pipeline.

        Returns
        -------
        list[Response]
            A list of model's responses.
        """
        completion_requests = [self.complete(m, params, metadata) for m in messages]
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
        generator = self.model_copy()
        generator.params = generator.params.model_copy(update=kwargs)
        return generator
