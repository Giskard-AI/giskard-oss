"""Adapt a Giskard BaseGenerator so garak's LLM-judge detectors can use it."""

import asyncio
import random
from typing import cast, override

from garak.attempt import Conversation, Message
from garak.generators.openai import OpenAICompatible
from giskard.agents import BaseGenerator
from giskard.llm.types import ChatMessageParam

from .._shared import await_on_loop


def _conversation_to_messages(
    conversation: Conversation | list[dict],  # pyright: ignore[reportMissingTypeArgument]
) -> list[ChatMessageParam]:
    """Convert a garak Conversation into Giskard chat-message dicts.

    Giskard's ``complete`` accepts ``ChatMessageParam`` dicts and validates them,
    so plain ``{\"role\", \"content\"}`` dicts are sufficient. Turns whose text is None
    (multimodal/file turns) are rendered as empty strings; the judge only sends
    system/user text turns in practice.
    """
    if isinstance(conversation, list):
        return cast(list[ChatMessageParam], conversation)

    return cast(
        list[ChatMessageParam],
        [
            {"role": turn.role, "content": turn.content.text or ""}
            for turn in conversation.turns
        ],
    )


class GiskardJudgeGenerator(OpenAICompatible):
    """Bridges garak judge detectors to a Giskard ``BaseGenerator``.

    Subclasses ``OpenAICompatible`` ONLY to satisfy ``judge.ModelAsJudge``'s
    ``isinstance`` gate. It deliberately skips ``OpenAICompatible.__init__`` (which
    builds a real ``openai.OpenAI`` client and requires an API key + model name) and
    makes ``_load_unsafe`` a no-op. ``generate()`` is inherited from garak's
    ``Generator`` and routes through the overridden ``_call_model`` below.
    """

    generator_family_name = "giskard"

    def __init__(  # pyright: ignore[reportMissingSuperCall] — no super().__init__() that constructs an OpenAI client.
        self,
        generator: BaseGenerator,
        loop: asyncio.AbstractEventLoop | None = None,
        name: str = "giskard-judge",
    ) -> None:
        self.name = name
        self.fullname = f"{self.generator_family_name} {name}"
        self._giskard = generator
        self._loop = loop
        self.client = None  # never used; _load_unsafe is a no-op
        self.generations = 1
        # garak Generator.generate() reads these on every call; we skip
        # Generator.__init__/_load_config to avoid the OpenAI client build and
        # API-key validation, so set the two attributes it actually needs here.
        self.seed = None
        self._rng = random.Random()

    @override
    def _load_unsafe(self) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        pass

    @override
    def _call_model(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        prompt: Conversation | list[dict],  # pyright: ignore[reportMissingTypeArgument]
        generations_this_call: int = 1,
    ) -> "list[Message | None]":
        messages = _conversation_to_messages(prompt)
        response = await_on_loop(self._giskard.complete(messages), self._loop)
        text = response.choices[0].message.text if response.choices else None
        return [Message(text=text) if text is not None else None]


def make_judge_detector(detector_cls, generator: BaseGenerator, loop):
    """Build a judge detector that scores with *generator* instead of its own LLM.

    ``judge.ModelAsJudge.__init__`` calls ``_load_generator`` (which loads an OpenAI
    generator by name and needs an API key). We subclass and override
    ``_load_generator`` to install a ``GiskardJudgeGenerator`` and set the token
    limit garak's ``judge_score`` reads.
    """
    from garak.resources.red_team.evaluation import get_token_limit

    class _GiskardJudge(detector_cls):  # type: ignore[valid-type, misc]
        def _load_generator(self) -> None:
            self.evaluation_generator = GiskardJudgeGenerator(generator, loop)
            self.evaluator_token_limit = get_token_limit(self.evaluation_generator.name)

    return _GiskardJudge()
