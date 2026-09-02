"""Internal helpers for streaming JSONL datasets and suite-scoped caching."""

import json
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

T = TypeVar("T")

_suite_cache: dict[tuple[object, ...], list[Any]] | None = None
_cache_lock = threading.Lock()


def iter_jsonl(
    path: Path | str, *, source: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-blank line in a JSONL file."""
    jsonl_path = Path(path)
    label = source if source is not None else str(jsonl_path)
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {label}:{line_num}: {exc}"
                ) from exc


def reservoir_sample[T](
    items: Iterable[T], k: int, rng: np.random.Generator
) -> list[T]:
    """Return a uniform random sample of size ``k`` from a stream (Algorithm R)."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return []

    reservoir: list[T] = []
    for index, item in enumerate(items):
        if index < k:
            reservoir.append(item)
        else:
            slot = int(rng.integers(0, index + 1))
            if slot < k:
                reservoir[slot] = item
    return reservoir


def activate_dataset_cache() -> None:
    global _suite_cache
    _suite_cache = {}


def deactivate_dataset_cache() -> None:
    global _suite_cache
    _suite_cache = None


def is_dataset_cache_active() -> bool:
    return _suite_cache is not None


def get_or_load_cached[T](
    key: tuple[object, ...], loader: Callable[[], list[T]]
) -> list[T]:
    if _suite_cache is None:
        return loader()

    with _cache_lock:
        cached = _suite_cache.get(key)
        if cached is not None:
            return cast(list[T], cached)
        loaded = loader()
        _suite_cache[key] = loaded
        return loaded
