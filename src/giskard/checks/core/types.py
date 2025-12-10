from collections.abc import AsyncGenerator, Awaitable, Callable, Generator

type SyncOrAsync[T] = T | Awaitable[T]
type SyncOrAsyncCallable[**P, R] = Callable[P, SyncOrAsync[R]]
type SyncOrAsyncGenerator[R, S] = Generator[R, S] | AsyncGenerator[R, S]
