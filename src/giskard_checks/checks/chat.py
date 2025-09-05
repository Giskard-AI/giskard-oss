from __future__ import annotations

from typing import Any, ClassVar, Literal

from counterpoint import Message
from counterpoint.chat import Content
from pydantic import Field

from giskard_checks.core.check import Check, CheckResult, CheckSeverity
from giskard_checks.interactions import ChatInteraction


def _extract_text_content(content: str | Content) -> str | None:
    if content is None:
        return None

    if isinstance(content, str):
        return content
    elif content.type == "text":
        return content.text

    return None


def _extract_text_contents(message: Message) -> list[str]:
    contents = (
        message.content if isinstance(message.content, list) else [message.content]
    )
    text_contents = [_extract_text_content(content) for content in contents]
    return [text_content for text_content in text_contents if text_content is not None]


def _extract_contents(
    output: list[Message], role: Literal["assistant", "user", "system"] = "assistant"
) -> list[str]:
    return [
        text_content
        for message in output
        if message.role == role
        for text_content in _extract_text_contents(message)
    ]


class StringMatchingCheck(Check[ChatInteraction]):
    KIND: ClassVar[str | None] = "string_matching"

    content: str = Field(..., description="The string to match in the output")
    role: Literal["assistant", "user", "system"] = Field(
        default="assistant",
        description="The role of the message to match in the output",
    )
    case_sensitive: bool = Field(
        default=True, description="Whether the string matching should be case sensitive"
    )
    severity: CheckSeverity = CheckSeverity.ERROR

    async def run(self, interaction: ChatInteraction) -> CheckResult:
        """Validate the presence (and optional name/args) of a tool call."""
        output = interaction.output
        if not output:
            return CheckResult.failure(
                kind=self.kind,
                name=self.name,
                description=self.description,
                message="No output messages to inspect for string matching",
                severity=self.severity,
                details={"reason": "empty_output"},
            )

        contents = _extract_contents(output, self.role)
        normalized_contents = contents
        match_content = self.content
        if not self.case_sensitive:
            normalized_contents = [content.lower() for content in contents]
            match_content = match_content.lower()

        matched = any(match_content in content for content in normalized_contents)
        if matched:
            return CheckResult.success(
                kind=self.kind,
                name=self.name,
                description=self.description,
                message="String matching succeeded",
                severity=self.severity,
            )

        return CheckResult.failure(
            kind=self.kind,
            name=self.name,
            description=self.description,
            message=f"String matching failed: {match_content} not found",
            severity=self.severity,
            details={
                "reason": "string_not_found",
                "match_content": self.content,
                "contents": contents,
                "role": self.role,
                "case_sensitive": self.case_sensitive,
            },
        )
