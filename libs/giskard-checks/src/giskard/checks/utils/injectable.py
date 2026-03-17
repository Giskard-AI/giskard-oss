import inspect
from collections.abc import AsyncGenerator
from typing import cast

from giskard.checks.core.types import GeneratorType, ProviderType, SyncOrAsyncGenerator
from giskard.checks.utils.generator import a_generator


def _validate_kwargs_keys[R](
    value_or_callable: ProviderType[..., R], kwargs_keys: set[str]
) -> tuple[list[str], set[str]]:
    if not callable(value_or_callable):
        return ([], set())

    signature = inspect.signature(value_or_callable)
    injected_positional_only_names: list[str] = []
    injected_kwarg_names: set[str] = set()
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.name in kwargs_keys:
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                injected_positional_only_names.append(param.name)
            else:
                injected_kwarg_names.add(param.name)
        else:
            default = cast(object, param.default)
            if default is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param.name}' is required but not in the injection requirements."
                )

    return injected_positional_only_names, injected_kwarg_names


class ValueProvider[**P, R]:
    _value_or_callable: ProviderType[..., R]
    _injected_positional_only_names: list[str]
    _injected_kwarg_names: set[str]

    def __init__(self, value_or_callable: ProviderType[..., R], kwargs_keys: set[str]):
        self._value_or_callable = value_or_callable
        (
            self._injected_positional_only_names,
            self._injected_kwarg_names,
        ) = _validate_kwargs_keys(value_or_callable, kwargs_keys)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # This is a static value
        if not callable(self._value_or_callable):
            return self._value_or_callable

        injected_positional_only_args = [
            kwargs[name]
            for name in self._injected_positional_only_names
            if name in kwargs
        ]
        injected_kwargs = {
            key: kwargs[key] for key in self._injected_kwarg_names if key in kwargs
        }

        result = self._value_or_callable(
            *injected_positional_only_args, **injected_kwargs
        )

        # Handle Awaitables
        if inspect.isawaitable(result):
            return await result

        return cast(R, result)


class ValueGenerator[**P, R, S]:
    _value_provider: ValueProvider[P, R | SyncOrAsyncGenerator[R, S]]

    def __init__(
        self, value_or_callable: GeneratorType[..., R, S], kwargs_keys: set[str]
    ):
        self._value_provider = cast(
            ValueProvider[P, R | SyncOrAsyncGenerator[R, S]],
            ValueProvider(value_or_callable, kwargs_keys),
        )

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, S]:
        value_or_generator = await self._value_provider(*args, **kwargs)
        return a_generator(value_or_generator)
