"""Adapt a Giskard generator so DeepTeam's simulator/evaluation LLM can use it.

``deepeval``/``DeepEvalBaseLLM`` are imported lazily so this module imports
without deepteam installed. ``GiskardDeepEvalLLM`` subclasses ``DeepEvalBaseLLM``
(satisfying DeepTeam's ``initialize_model`` acceptance) and routes both sync and
async generation through a Giskard ``BaseGenerator``.

DeepEval treats our wrapper as a *non-native* model and calls
``(a_)generate(prompt, schema=SomePydanticModel)`` expecting a validated instance
of that schema back (it then reads attributes such as ``res.data`` off it). We
honour that by routing the schema through the generator's structured-output path
(``chat(prompt).with_output(schema).run()``), which sets ``response_format`` on
the underlying LLM call and validates/retries the reply into the schema. Without
a schema we return raw text via ``complete``.
"""

import asyncio
from typing import Any

from .._shared import await_on_loop


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

        async def a_generate(
            self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            # DeepEval passes a Pydantic ``schema`` for structured calls (attack
            # simulation, evaluation verdicts) and reads attributes off the
            # returned instance (e.g. ``res.data``). Route the schema through the
            # generator's structured-output workflow so the underlying LLM gets a
            # real ``response_format`` and the reply is validated into ``schema``.
            if schema is not None:
                chat = await self._giskard.chat(prompt).with_output(schema).run()
                return chat.output
            messages = [{"role": "user", "content": prompt}]
            response = await self._giskard.complete(messages)
            return response.choices[0].message.text if response.choices else ""

        def generate(
            self, prompt: str, schema: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            return await_on_loop(self.a_generate(prompt, schema), self._loop)

        def get_model_name(self) -> str:
            return "giskard-deepteam"

    return GiskardDeepEvalLLM(generator, loop)
