from collections.abc import AsyncGenerator
from typing import Any, override
from unittest.mock import MagicMock, patch

from ..core import Trace
from ..core.interaction import Interaction, InteractionSpec


def _call_to_dict(call: Any) -> dict[str, Any]:
    """Serialise a ``unittest.mock._Call`` to a JSON-serialisable dict.

    ``MagicMock`` stores raw ``_Call`` objects (a private ``tuple`` subclass)
    in ``call_args``, ``call_args_list`` and ``mock_calls``. Pydantic's JSON
    mode serializer (used by ``Trace.model_dump(mode="json")`` and the prompt
    ``| fence`` filter) cannot serialise ``_Call``, so any check that renders
    ``{{ trace | fence }}`` (e.g. ``Conformity`` / ``LLMJudge``) raised
    ``TypeError: '_Call' object is not an instance of 'SchemaSerializer'``
    when paired with ``WithSpy`` (gap G-08b).

    The dict keeps the same ``args`` / ``kwargs`` access shape, so existing
    JSONPath targets such as ``metadata[...]['call_args'].args[0]`` keep working
    (``jsonpath-ng`` reads the ``args`` key of the dict).
    """
    return {"args": list(call.args), "kwargs": dict(call.kwargs)}


@InteractionSpec.register("with_spy")
class WithSpy[InputType, OutputType, TraceType: Trace](  # pyright: ignore[reportMissingTypeArgument]
    InteractionSpec[InputType, OutputType, TraceType]
):
    interaction_generator: InteractionSpec[InputType, OutputType, TraceType]
    target: str

    @override
    async def generate(
        self, trace: TraceType
    ) -> AsyncGenerator[Interaction[InputType, OutputType], TraceType]:
        spy = MagicMock()
        with patch(self.target, spy):
            generator = self.interaction_generator.generate(trace)

            try:
                trace = yield self._patch_interaction(await anext(generator), spy)
                while True:
                    trace = yield self._patch_interaction(
                        await generator.asend(trace), spy
                    )
            except StopAsyncIteration:
                return
            finally:
                await generator.aclose()

    def _patch_interaction(
        self, interaction: Interaction[InputType, OutputType], spy: MagicMock
    ) -> Interaction[InputType, OutputType]:
        interaction = Interaction(
            inputs=interaction.inputs,
            outputs=interaction.outputs,
            metadata=interaction.metadata
            | {
                self.target: {
                    "call_args_list": [
                        _call_to_dict(call) for call in spy.call_args_list
                    ],
                    "call_count": spy.call_count,
                    "call_args": (
                        _call_to_dict(spy.call_args)
                        if spy.call_args is not None
                        else None
                    ),
                    "mock_calls": [
                        {"name": call[0], **_call_to_dict(call)}
                        for call in spy.mock_calls
                    ],
                }
            },
        )
        spy.reset_mock()
        return interaction
