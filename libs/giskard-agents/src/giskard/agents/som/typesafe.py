"""TypeSafe's native System One API transport."""

import os
from collections.abc import Sequence
from typing import Literal, override

import httpx
from giskard.llm.types import ChatMessage, Usage
from pydantic import BaseModel, Field, field_validator

from .base import BaseSOM, SOMResponse


class _NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class _Answers(BaseModel):
    decision: _NoulAnswer


class _TokenUsage(BaseModel):
    input_tokens: int = Field(strict=True, ge=0)
    output_tokens: int = Field(strict=True, ge=0)


class _SystemOneResponse(BaseModel):
    model: str = Field(min_length=1)
    answers: _Answers
    usage: _TokenUsage


@BaseSOM.register("typesafe")
class TypeSafeSOM(BaseSOM):
    """Predict a probability with TypeSafe's native API.

    ``model`` is the native model name; ``jev`` expands to ``jev-latest``.
    The API endpoint is configured locally through ``TYPESAFE_BASE_URL`` or
    ``TYPESAFE_API_BASE`` and defaults to TypeSafe's public API.

    A LiteLLM pass-through base such as ``http://localhost:4000/typesafe``
    also works. Credentials are read from ``TYPESAFE_API_KEY`` only when making
    a request and are never stored in the model. Keeping transport configuration
    out of the model prevents serialized scenarios from selecting a destination
    or a different environment variable containing a secret.
    """

    @field_validator("model")
    @classmethod
    def _expand_alias(cls, model: str) -> str:
        return "jev-latest" if model == "jev" else model

    def _endpoint(self) -> str:
        base = (
            os.environ.get("TYPESAFE_BASE_URL")
            or os.environ.get("TYPESAFE_API_BASE")
            or "https://api.typesafe.ai"
        ).rstrip("/")
        if base.endswith("/v1/systemone"):
            return base
        return base + ("/systemone" if base.endswith("/v1") else "/v1/systemone")

    @override
    async def predict(
        self,
        messages: Sequence[ChatMessage],
        question: str,
        *,
        timeout: float | int | None = None,
    ) -> SOMResponse:
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("Set TYPESAFE_API_KEY to use the TypeSafe SOM provider.")

        async with httpx.AsyncClient(
            timeout=timeout if timeout is not None else 30.0
        ) as client:
            response = await client.post(
                self._endpoint(),
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "state": [
                        message.model_dump(mode="json", exclude_none=True)
                        for message in messages
                    ],
                    "questions": {
                        "decision": {"type": "noul", "instructions": question}
                    },
                },
            )
        response.raise_for_status()
        payload = _SystemOneResponse.model_validate(response.json())
        return SOMResponse(
            probability=payload.answers.decision.noul,
            model=payload.model,
            usage=Usage(
                input_tokens=payload.usage.input_tokens,
                output_tokens=payload.usage.output_tokens,
                total_tokens=payload.usage.input_tokens + payload.usage.output_tokens,
            ),
        )
