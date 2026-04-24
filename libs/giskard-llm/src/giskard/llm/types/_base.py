import json
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator


class _BaseModel(BaseModel):
    """Shared base for all giskard-llm response models. Defaults model_dump to exclude None fields."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


ArgumentDict = Annotated[dict[str, object], BeforeValidator(_coerce_json)]
