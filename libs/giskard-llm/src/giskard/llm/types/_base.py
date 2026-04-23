from typing import Any

from pydantic import BaseModel


class _BaseModel(BaseModel):
    """Shared base for all giskard-llm response models. Defaults model_dump to exclude None fields."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)
