import copy
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
            # Copy rather than call the constructor: dict subclasses such as
            # defaultdict do not accept a mapping as their first argument.
            normalized = copy.copy(data)
            for k, v in data.items():
                normalized[k] = normalize_data(v, normalization_form)
            return normalized
        case tuple() if hasattr(data, "_fields"):
            # Named tuples take their fields as positional arguments.
            return type(data)(*(normalize_data(v, normalization_form) for v in data))
        case list() | tuple() | set():
            return type(data)(normalize_data(v, normalization_form) for v in data)
        case str():
            return normalize_string(data, normalization_form)
        case _:
            return data
