"""Adapt a Giskard generator so DeepTeam's simulator/evaluation LLM can use it.

``deepeval``/``DeepEvalBaseLLM`` are imported lazily so this module imports
without deepteam installed. ``GiskardDeepEvalLLM`` subclasses ``DeepEvalBaseLLM``
(satisfying DeepTeam's ``initialize_model`` acceptance) and routes both sync and
async generation through a Giskard ``BaseGenerator.complete``.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any


def _await_on_loop[T](
    coro: Coroutine[Any, Any, T],
    loop: "asyncio.AbstractEventLoop | None",
) -> T:
    """Run *coro* to completion. If *loop* is a running loop on another thread,
    schedule there via ``run_coroutine_threadsafe``; otherwise ``asyncio.run``.

    DeepTeam may call ``generate`` (sync) from inside its own async machinery on
    a worker thread; a nested ``asyncio.run`` there would break shared locks in
    the Giskard generator, so we reuse the scan's loop when one is provided.
    """
    if loop is None:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def make_deepeval_llm(
    generator: Any, loop: "asyncio.AbstractEventLoop | None" = None
) -> Any:
    """Return a ``GiskardDeepEvalLLM`` wrapping *generator* (a giskard BaseGenerator)."""
    from deepeval.models import DeepEvalBaseLLM

    class GiskardDeepEvalLLM(DeepEvalBaseLLM):
        """Routes DeepTeam LLM calls through a Giskard generator."""

        def __init__(self, giskard_generator: Any, event_loop: Any) -> None:
            self._giskard = giskard_generator
            self._loop = event_loop
            super().__init__()

        def load_model(self) -> Any:
            return self._giskard

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            messages = [{"role": "user", "content": prompt}]
            response = await self._giskard.complete(messages)
            return response.choices[0].message.text if response.choices else ""

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            return _await_on_loop(self.a_generate(prompt), self._loop)

        def get_model_name(self) -> str:
            return "giskard-deepteam"

    return GiskardDeepEvalLLM(generator, loop)
