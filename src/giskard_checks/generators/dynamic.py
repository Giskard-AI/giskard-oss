from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Callable, TypeGuard, cast

from pydantic import Field

from ..core.context import Context
from ..core.interaction_result import InteractionResult
from .base import InteractionGenerator

ResultOrAwaitable = Awaitable[InteractionResult[Any, Any]] | InteractionResult[Any, Any]

SingleArgCallable = Callable[[], ResultOrAwaitable]
ContextArgCallable = Callable[[Context], ResultOrAwaitable]


def is_single_arg_sync_generator(
    value: SingleArgCallable | ContextArgCallable,
) -> TypeGuard[SingleArgCallable]:
    sig = inspect.signature(value)
    return len(sig.parameters) == 0


@InteractionGenerator.register("dynamic")
class DynamicInteraction(InteractionGenerator):
    """A dynamic interaction generator that uses a callable to generate interactions.

    This generator accepts a callable (sync or async) that can either:
    - Take no parameters: `callable() -> InteractionResult`
    - Take a Context parameter: `callable(context: Context) -> InteractionResult`

    The callable is automatically detected based on its signature and called
    appropriately. The callable must return an `InteractionResult`.

    Note: The `fn` field is not serializable and will not be included in
    serialization. As a result, `DynamicInteraction` instances cannot be reliably
    serialized/deserialized. This is intended for programmatic/test use only.

    Examples
    --------
    >>> # Sync callable without context
    >>> def simple_generator():
    ...     return InteractionResult(inputs="hello", outputs="world")
    >>> interaction = DynamicInteraction(fn=simple_generator)

    >>> # Async callable with context
    >>> async def context_aware_generator(context: Context):
    ...     return InteractionResult(inputs=context.previous_interactions, outputs="response")
    >>> interaction = DynamicInteraction(fn=context_aware_generator)
    """

    fn: SingleArgCallable | ContextArgCallable = Field(
        exclude=True,
        repr=False,
        description="Function to execute for generating interactions. Not serializable.",
    )

    async def generate(self, context: Context) -> InteractionResult[Any, Any]:
        """Generate an interaction by calling the provided function.

        Parameters
        ----------
        context : Context
            The context containing previous interactions and metadata.

        Returns
        -------
        InteractionResult[Any, Any]
            The interaction result returned by the callable.

        Raises
        ------
        TypeError
            If the callable doesn't return an InteractionResult.
        """
        # Detect if the callable expects a Context parameter
        if is_single_arg_sync_generator(self.fn):
            result = self.fn()
        else:
            context_fn = cast(ContextArgCallable, self.fn)
            result = context_fn(context)

        # Handle async callables
        if inspect.isawaitable(result):
            result = await result

        # Validate return type
        if not isinstance(result, InteractionResult):
            raise TypeError(
                f"DynamicInteraction callable must return InteractionResult, but got {type(result).__name__}: {result}"
            )

        return result

    @classmethod
    def from_callable(
        cls, callable: Callable[..., Any | Awaitable[Any]], *args, **kwargs
    ) -> DynamicInteraction:
        async def wrapped() -> InteractionResult[Any, Any]:
            result = callable(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result

            return InteractionResult(
                inputs={"args": args, "kwargs": kwargs}, outputs=result
            )

        return cls(fn=wrapped)
