from counterpoint.chat import Message
from counterpoint.generators.base import BaseGenerator, GenerationParams, Response
from pydantic import Field


class MockGenerator(BaseGenerator):
    output: str | None
    calls: list[tuple[list[Message], GenerationParams | None]] = Field(
        default_factory=list
    )

    def with_output(self, output: str | None) -> "MockGenerator":
        self.output = output
        return self

    def clear_calls(self) -> None:
        self.calls = []

    async def _complete(
        self, messages: list[Message], params: GenerationParams | None = None
    ) -> Response:
        self.calls.append((messages, params))
        return Response(
            message=Message(role="assistant", content=self.output), finish_reason="stop"
        )
