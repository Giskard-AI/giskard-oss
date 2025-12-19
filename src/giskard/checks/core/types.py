from collections.abc import AsyncGenerator, Awaitable, Callable, Generator

type SyncOrAsync[T] = T | Awaitable[T]
type SyncOrAsyncCallable[**P, R] = Callable[P, SyncOrAsync[R]]
type SyncOrAsyncGenerator[R, S] = Generator[R, S] | AsyncGenerator[R, S]
type GeneratorProvider[**P, R, S] = Callable[P, SyncOrAsyncGenerator[R, S]]

type ProviderType[**P, T] = T | SyncOrAsyncCallable[P, T]
type GeneratorType[**P, R, S] = ProviderType[P, R] | GeneratorProvider[P, R, S]
