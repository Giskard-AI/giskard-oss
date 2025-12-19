import inspect
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

from ..core.types import GeneratorType, ProviderType, SyncOrAsync, SyncOrAsyncGenerator
from ..utils.generator import a_generator
from .parameter_injection import (
    CallableInjectionMapping,
    ParameterInjectionRequirement,
)


class ValueProvider[**P, R]:
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError

    @classmethod
    def from_mapping(
        cls,
        value_or_callable: ProviderType[..., R],
        *args_reqs: ParameterInjectionRequirement,
        **kwargs_reqs: ParameterInjectionRequirement,
    ) -> "ValueProvider[P, R]":
        if isinstance(value_or_callable, Callable):
            return CallableValueProvider(
                callable=value_or_callable,
                injection_mapping=CallableInjectionMapping.from_callable(
                    value_or_callable, *args_reqs, **kwargs_reqs
                ),
            )

        return StaticValueProvider(value=value_or_callable)


@dataclass(frozen=True)
class StaticValueProvider[**P, R](ValueProvider[P, R]):
    value: R

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.value


@dataclass(frozen=True)
class CallableValueProvider[**P, R](ValueProvider[P, R]):
    callable: Callable[..., SyncOrAsync[R]]
    injection_mapping: CallableInjectionMapping

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        maybe_awaitable = self.injection_mapping.inject_parameters(
            self.callable, *args, **kwargs
        )()
        if inspect.isawaitable(maybe_awaitable):
            return await maybe_awaitable

        return maybe_awaitable


@dataclass(frozen=True)
class ValueGeneratorProvider[**P, R, S]:
    provider: ValueProvider[P, R | SyncOrAsyncGenerator[R, S]]

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, S]:
        value_or_generator = await self.provider(*args, **kwargs)
        return a_generator(value_or_generator)

    @classmethod
    def from_mapping(
        cls,
        value_or_callable: GeneratorType[P, R, S],
        *args_reqs: ParameterInjectionRequirement,
        **kwargs_reqs: ParameterInjectionRequirement,
    ) -> "ValueGeneratorProvider[P, R, S]":
        return cls(
            provider=ValueProvider.from_mapping(
                value_or_callable, *args_reqs, **kwargs_reqs
            ),
        )
