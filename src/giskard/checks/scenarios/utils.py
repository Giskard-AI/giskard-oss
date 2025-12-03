import inspect
from collections.abc import AsyncGenerator, Awaitable, Generator
from typing import Callable, cast

from typing_extensions import TypeGuard

from ..core.trace import Trace
from ..utils.generator import a_generator, map_a_generator


def _is_with_trace_callable[ReturnType, InputType, OutputType](
    callable: Callable[[], ReturnType]
    | Callable[[Trace[InputType, OutputType]], ReturnType],
) -> TypeGuard[Callable[[Trace[InputType, OutputType]], ReturnType]]:
    signature = inspect.signature(callable)
    if len(signature.parameters) != 1:
        return False

    return True


def _infer_trace_argument_type[ReturnType, InputType, OutputType](
    callable: Callable[[Trace[InputType, OutputType]], ReturnType],
) -> type[Trace[InputType, OutputType]]:
    signature = inspect.signature(callable)

    param = next(iter(signature.parameters.values()))

    return param.annotation or Trace


async def execute_code[ReturnType, InputType, OutputType](
    value_or_callable: ReturnType
    | Callable[[], ReturnType | Awaitable[ReturnType]]
    | Callable[[Trace[InputType, OutputType]], ReturnType | Awaitable[ReturnType]],
    trace: Trace[InputType, OutputType],
) -> ReturnType:
    """Execute a value or callable, handling both sync and async cases.

    This function supports three types of inputs:
    1. A static value: returned as-is
    2. A callable that accepts a Trace parameter: called with the provided trace
    3. A callable that takes no parameters: called without arguments

    All callables may return either a value directly or an awaitable (coroutine).

    Args:
        value_or_callable: Either a static value of type ReturnType, or a callable
            that may accept a Trace parameter and returns ReturnType (sync or async).
        trace: The trace object to pass to callables that accept it.

    Returns:
        The value of type ReturnType, either from the input directly or from
        executing the callable (awaiting if necessary).
    """
    if not isinstance(value_or_callable, Callable):
        return value_or_callable

    trace_argument_type = None
    if _is_with_trace_callable(value_or_callable):
        trace_argument_type = _infer_trace_argument_type(value_or_callable)
        value_or_awaitable = value_or_callable(
            trace_argument_type(interactions=trace.interactions)
        )

        if isinstance(value_or_awaitable, Generator | AsyncGenerator):
            generator = cast(
                Generator[ReturnType, Trace[InputType, OutputType], None]
                | AsyncGenerator[ReturnType, Trace[InputType, OutputType]],
                value_or_awaitable,
            )
            value_or_awaitable = map_a_generator(
                a_generator(generator),
                map_send=lambda x: trace_argument_type(interactions=x.interactions),
            )
    else:
        value_or_awaitable = cast(Callable[[], ReturnType], value_or_callable)()

    if inspect.isawaitable(value_or_awaitable):
        return await value_or_awaitable

    return cast(ReturnType, value_or_awaitable)


async def make_generator[ReturnType, InputType, OutputType](
    value_or_callable: ReturnType
    | Callable[
        [],
        ReturnType
        | AsyncGenerator[ReturnType, Trace[InputType, OutputType]]
        | Generator[ReturnType, Trace[InputType, OutputType], None],
    ]
    | Callable[
        [Trace[InputType, OutputType]],
        ReturnType
        | AsyncGenerator[ReturnType, Trace[InputType, OutputType]]
        | Generator[ReturnType, Trace[InputType, OutputType], None],
    ],
    trace: Trace[InputType, OutputType],
) -> AsyncGenerator[ReturnType, Trace[InputType, OutputType]]:
    result = await execute_code(value_or_callable, trace)

    return a_generator(result)
