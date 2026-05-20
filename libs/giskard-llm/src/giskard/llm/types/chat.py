from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

from ._base import ArgumentDict, _BaseModel

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

# -- Utility functions -------------------------------------------------------------


class _TextContentProtocol(Protocol):
    text: str | None


class _TextualizableContentProtocol(Protocol):
    @property
    def text(self) -> str | None: ...


def _extract_text(
    content: str
    | Sequence[_TextualizableContentProtocol | _TextContentProtocol]
    | None,
) -> str | None:
    if isinstance(content, str) or content is None:
        return content

    texts = [c.text for c in content if c.text is not None]

    return "\n".join(texts) if texts else None


_EMPTY_RENDER_TEXT = "[dim italic]empty[/dim italic]"

# -- Chat content types -------------------------------------------------------------


class TextContent(_BaseModel):
    type: Literal["text"] = "text"
    text: str


class RefusalContent(_BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str

    @property
    def text(self) -> str:
        return self.refusal


CompletionContent = TextContent | RefusalContent

# -- Chat Message types -------------------------------------------------------------


class ToolCallFunction(_BaseModel):
    name: str
    arguments: ArgumentDict


class ToolCall(_BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: ToolCallFunction


class SystemMessage(_BaseModel):
    role: Literal["system"] = "system"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel

        yield Panel(
            self.text or _EMPTY_RENDER_TEXT,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="left",
            border_style="grey37",
            padding=(1, 2),
        )


class DeveloperMessage(_BaseModel):
    role: Literal["developer"] = "developer"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel

        yield Panel(
            self.text or _EMPTY_RENDER_TEXT,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="left",
            border_style="red3",
            padding=(1, 2),
        )


class UserMessage(_BaseModel):
    role: Literal["user"] = "user"
    content: str | Sequence[TextContent]

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel

        yield Panel(
            self.text or _EMPTY_RENDER_TEXT,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="right",
            border_style="bright_blue",
            padding=(1, 2),
        )


class AssistantMessage(_BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | Sequence[CompletionContent] | None = None
    refusal: str | None = None
    tool_calls: Sequence[ToolCall] | None = None

    @property
    def text(self) -> str | None:
        texts = [
            text
            for text in (self.refusal, _extract_text(self.content))
            if text is not None
        ]

        return "\n".join(texts) if texts else None

    @property
    def is_refusal(self) -> bool:
        return self.refusal is not None or (
            isinstance(self.content, Sequence)
            and any(isinstance(c, RefusalContent) for c in self.content)
        )

    @property
    def transcript(self) -> str:
        message = self.text or "empty"
        if self.tool_calls is not None:
            for tool_call in self.tool_calls:
                message += f"\n>[tool_call:{tool_call.function.name}:{tool_call.id}]: {tool_call.function.arguments}"

        return f"[{self.role}]: {message}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel
        from rich.text import Text

        content = Text()
        if self.text:
            content.append(self.text)

        if self.tool_calls:
            for tc in self.tool_calls:
                content.append("\n\n")
                tool_text = Text.assemble(
                    ("🔧 Tool Call: ", "bold cyan"),
                    (f"{tc.function.name}", "italic cyan"),
                    (f"\nArguments: {tc.function.arguments}", "grey70"),
                )
                content.append(tool_text)

        if not self.text and not self.tool_calls:
            content.append(_EMPTY_RENDER_TEXT)

        yield Panel(
            content,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="left",
            border_style="red" if self.is_refusal else "orchid",
            padding=(1, 2),
        )


class ToolMessage(_BaseModel):
    role: Literal["tool"] = "tool"
    content: str | Sequence[TextContent]
    tool_call_id: str

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel

        yield Panel(
            self.text or _EMPTY_RENDER_TEXT,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="left",
            border_style="green3",
            padding=(1, 2),
        )


class FunctionMessage(_BaseModel):
    content: str | None = None
    name: str
    role: Literal["function"] = "function"

    @property
    def text(self) -> str | None:
        return _extract_text(self.content)

    @property
    def transcript(self) -> str:
        return f"[{self.role}]: {self.text or 'empty'}"

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        from rich.panel import Panel

        yield Panel(
            self.text or _EMPTY_RENDER_TEXT,
            title=f"[bold]{self.role.upper()}[/bold]",
            title_align="left",
            border_style="green3",
            padding=(1, 2),
        )


ChatMessage = (
    SystemMessage
    | DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | FunctionMessage
)
