import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import giskard.scan.generators.base as base_mod
import numpy as np
from giskard.scan.generators.base import LocalDatasetScenarioGenerator, ScenarioContext
from giskard.scan.utils import dataset_loader as dl_mod
from giskard.scan.utils.dataset_loader import (
    activate_dataset_cache,
    deactivate_dataset_cache,
)


class _StubDatasetGenerator(LocalDatasetScenarioGenerator):
    dataset_name: str = "stub"


def _write_jsonl(path: Path, count: int) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"name": f"s{i}", "steps": [], "annotations": {}})
            for i in range(count)
        )
        + "\n",
        encoding="utf-8",
    )


async def test_streaming_subsample_returns_requested_count(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "stub.jsonl", 500)
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    gen = _StubDatasetGenerator()
    result = await gen.generate_scenario(
        ScenarioContext(description="desc", languages=["en"]),
        max_scenarios=20,
        rng=np.random.default_rng(42),
    )
    assert len(result) == 20


async def test_streaming_subsample_uses_reservoir_with_budget(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "stub.jsonl", 500)
    monkeypatch.setattr(base_mod, "_DATA_DIR", tmp_path)
    seen = {"k": None}
    original = dl_mod.reservoir_sample

    def spy(items, k, rng):
        seen["k"] = k
        return original(items, k, rng)

    monkeypatch.setattr(base_mod, "reservoir_sample", spy)
    gen = _StubDatasetGenerator()
    await gen.generate_scenario(
        ScenarioContext(description="desc", languages=["en"]),
        max_scenarios=20,
        rng=np.random.default_rng(42),
    )
    assert seen["k"] == 20


async def test_suite_cache_shares_single_harmbench_parse(tmp_path, monkeypatch):
    from giskard.scan.generators import huggingface as hf_mod
    from giskard.scan.generators.gcg import GCGInjectionScenarioGenerator
    from giskard.scan.generators.huggingface import HuggingFaceDatasetScenarioGenerator

    files: dict[str, str] = {}
    configs: list[dict[str, Any]] = []
    repo_file = "harmbench.en.jsonl"
    path = tmp_path / repo_file
    path.write_text(
        "\n".join(
            json.dumps({"name": f"s{i}", "steps": [], "annotations": {}})
            for i in range(30)
        )
        + "\n",
        encoding="utf-8",
    )
    files[repo_file] = str(path)
    configs.append(
        {
            "config_name": "en",
            "data_files": [{"split": "test", "path": repo_file}],
        }
    )

    monkeypatch.setattr(
        hf_mod.DatasetCard,
        "load",
        staticmethod(
            lambda repo_id, repo_type=None: SimpleNamespace(
                data=SimpleNamespace(configs=list(configs))
            )
        ),
    )
    monkeypatch.setattr(
        hf_mod, "list_repo_files", lambda repo_id, repo_type=None: list(files)
    )
    monkeypatch.setattr(
        hf_mod,
        "hf_hub_download",
        lambda repo_id, repo_file, repo_type=None: files[repo_file],
    )
    hf_mod._language_subsets.cache_clear()

    load_calls = {"count": 0}
    original_iter = dl_mod.iter_jsonl

    def spy_iter(path, source=None):
        load_calls["count"] += 1
        return original_iter(path, source=source)

    monkeypatch.setattr(hf_mod, "iter_jsonl", spy_iter)

    repo_id = "giskardai/harmbench-scenarios"
    plain = HuggingFaceDatasetScenarioGenerator(repo_id=repo_id)
    gcg = GCGInjectionScenarioGenerator()
    context = ScenarioContext(description="desc", languages=["en"])
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)

    activate_dataset_cache()
    try:
        await plain.generate_scenario(context, max_scenarios=5, rng=rng_a)
        await gcg.generate_scenario(context, max_scenarios=5, rng=rng_b)
    finally:
        deactivate_dataset_cache()

    assert load_calls["count"] == 1
