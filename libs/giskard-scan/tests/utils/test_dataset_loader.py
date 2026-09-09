from pathlib import Path

import numpy as np
import pytest
from giskard.scan.utils.dataset_loader import iter_jsonl, reservoir_sample


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_raises_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(iter_jsonl(path))


def test_reservoir_sample_returns_all_when_stream_shorter_than_k() -> None:
    rng = np.random.default_rng(0)
    assert reservoir_sample([1, 2, 3], 10, rng) == [1, 2, 3]


def test_reservoir_sample_size_matches_k() -> None:
    rng = np.random.default_rng(42)
    sample = reservoir_sample(range(500), 20, rng)
    assert len(sample) == 20


def test_reservoir_sample_is_reproducible_with_same_seed() -> None:
    sample_a = reservoir_sample(range(500), 20, np.random.default_rng(7))
    sample_b = reservoir_sample(range(500), 20, np.random.default_rng(7))
    assert sample_a == sample_b


def test_reservoir_sample_rejects_negative_k() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reservoir_sample([1], -1, np.random.default_rng(0))
