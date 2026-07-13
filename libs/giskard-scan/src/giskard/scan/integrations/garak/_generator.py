import asyncio
import uuid
from typing import override

from garak.attempt import Conversation, Message
from garak.generators.base import Generator
from giskard.checks import DatasetInputGenerator, Interact, Target, Trace

from .._shared import await_on_loop
from ._bridge import _conv_uuid


class TargetGenerator[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    Generator
):
    """Drives a Giskard ``Target`` as a garak generator.

    Garak calls ``_call_model`` once per turn with the conversation so far and has
    no notion of our ``Trace``. We key the trace by garak's conversation uuid in
    ``internal_cache`` so a multi-turn probe accumulates one linear trace across
    its turns, and pop on read so finished conversations don't leak.
    """

    generator_family_name = "target"
    target: Target[InputType, OutputType, TraceType]
    internal_cache: dict[str, TraceType]
    _loop: asyncio.AbstractEventLoop | None

    def __init__(
        self,
        target: Target[InputType, OutputType, TraceType],
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(name="target-generator")
        self.target = target
        self.internal_cache = dict()
        self._loop = loop
        # Cache one empty trace: Trace.for_target inspects the target's type hints
        # (get_type_hints), which we don't want to repeat on every turn. Trace is
        # frozen, so a single empty instance is safe to share as the default —
        # with_interaction returns a new trace rather than mutating this one.
        self._empty_trace: TraceType = Trace.for_target(target)

    def get_trace(self, conversation: Conversation) -> TraceType:
        """Pop on read to clean up the cache."""
        conv_uuid = _conv_uuid(conversation)
        if conv_uuid is None:
            return self._empty_trace
        return self.internal_cache.pop(conv_uuid, self._empty_trace)

    @override
    def _call_model(  # pyright: ignore[reportGeneralTypeIssues]
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> list[Message | None]:
        conv_uuid = _conv_uuid(prompt) or str(uuid.uuid4())
        trace = self.internal_cache.pop(conv_uuid, self._empty_trace)

        # Garak allows Message.text=None for multimodal/file turns. DatasetInputGenerator
        # requires a non-empty str, so skip target invocation and let detectors score None.
        text = prompt.last_message().text
        if not text:
            self.internal_cache[conv_uuid] = trace
            return [Message(notes={"uuid": conv_uuid})]

        # DatasetInputGenerator carries the garak prompt into the target input:
        # verbatim for str targets, or mapped into the target's schema (LLM-resolved)
        # for structured InputTypes. Keep it so structured targets keep working.
        interaction = Interact[InputType, OutputType, TraceType](
            inputs=DatasetInputGenerator(prompt=text),
            outputs=self.target,
        )

        trace = await_on_loop(trace.with_interaction(interaction), self._loop)

        # Trace is frozen — with_interaction returns a NEW trace, so write it back
        # under the same uuid for the next turn in this conversation to build on.
        self.internal_cache[conv_uuid] = trace

        assert trace.last is not None, "Trace last is None"
        outputs = trace.last.outputs
        # Message.text is typed as str (default None at runtime). Match the judge
        # generator: omit text when the target returned nothing.
        if outputs is None:
            return [Message(notes={"uuid": conv_uuid})]
        return [Message(text=str(outputs), notes={"uuid": conv_uuid})]
