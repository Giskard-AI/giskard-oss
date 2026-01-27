import re
import unicodedata
from typing import Literal

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


def normalize_string(
    value: str, normalization_form: NormalizationForm | None = None
) -> str:
    """Normalize a string using the given normalization form."""
    if normalization_form is not None:
        value = unicodedata.normalize(normalization_form, value)

    # Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
    # and trim leading/trailing whitespace
    return re.sub(r"\s+", " ", value).strip()


def normalize_data[T](
    data: T, normalization_form: NormalizationForm | None = None
) -> T:
    """Normalize a dictionary or list using the given normalization form."""

    match data:
        case dict():
            return type(data)(
                {k: normalize_data(v, normalization_form) for k, v in data.items()}
            )
        case list() | tuple() | set():
            return type(data)(normalize_data(v, normalization_form) for v in data)
        case str():
            return normalize_string(data, normalization_form)
        case _:
            return data
