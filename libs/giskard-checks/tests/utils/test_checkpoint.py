"""Tests for RunStore checkpoint append / load / resume fingerprinting."""

import asyncio
import json
from pathlib import Path

import pytest
from giskard.checks.utils.checkpoint import (
    CheckpointFingerprintError,
    RunStore,
    resolve_checkpoint_options,
    store_path_for,
)


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


async def test_append_and_list_completed_ids(checkpoint_dir: Path) -> None:
    store = await RunStore.open(
        checkpoint_dir,
        fingerprint={"phase": "run", "suite": "demo"},
        resume=False,
    )
    await store.append(
        "scenario_finished",
        id="sc-1",
        payload={"status": "pass"},
    )
    await store.append(
        "scenario_finished",
        id="sc-2",
        payload={"status": "fail"},
    )

    assert store.completed_ids("scenario_finished") == {"sc-1", "sc-2"}
    assert store.load_payloads("scenario_finished")["sc-1"] == {"status": "pass"}


async def test_concurrent_appends_are_valid_jsonl(checkpoint_dir: Path) -> None:
    store = await RunStore.open(
        checkpoint_dir,
        fingerprint={"phase": "run"},
        resume=False,
    )

    async def write_one(i: int) -> None:
        await store.append(
            "scenario_finished",
            id=f"sc-{i}",
            payload={"i": i},
        )

    await asyncio.gather(*(write_one(i) for i in range(20)))

    lines = (checkpoint_dir / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 20
    for line in lines:
        record = json.loads(line)
        assert record["type"] == "scenario_finished"
        assert "id" in record
        assert "payload" in record
    assert store.completed_ids("scenario_finished") == {f"sc-{i}" for i in range(20)}


async def test_resume_loads_existing_events(checkpoint_dir: Path) -> None:
    fingerprint = {"phase": "run", "suite": "demo"}
    store = await RunStore.open(checkpoint_dir, fingerprint=fingerprint, resume=False)
    await store.append("scenario_finished", id="sc-1", payload={"ok": True})

    resumed = await RunStore.open(checkpoint_dir, fingerprint=fingerprint, resume=True)
    assert resumed.completed_ids("scenario_finished") == {"sc-1"}


async def test_fingerprint_mismatch_raises(checkpoint_dir: Path) -> None:
    await RunStore.open(
        checkpoint_dir,
        fingerprint={"phase": "run", "suite": "a"},
        resume=False,
    )
    with pytest.raises(CheckpointFingerprintError):
        await RunStore.open(
            checkpoint_dir,
            fingerprint={"phase": "run", "suite": "b"},
            resume=True,
        )


async def test_fingerprint_force_allows_mismatch(checkpoint_dir: Path) -> None:
    await RunStore.open(
        checkpoint_dir,
        fingerprint={"phase": "run", "suite": "a"},
        resume=False,
    )
    store = await RunStore.open(
        checkpoint_dir,
        fingerprint={"phase": "run", "suite": "b"},
        resume="force",
    )
    assert store.path == checkpoint_dir


def test_resolve_auto_uses_fingerprint_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GISKARD_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("GISKARD_CHECKPOINT_RESUME", raising=False)
    monkeypatch.delenv("GISKARD_CHECKPOINT", raising=False)
    monkeypatch.chdir(tmp_path)
    fingerprint = {"phase": "run", "suite": "demo"}
    path, resume = resolve_checkpoint_options(fingerprint=fingerprint)
    assert resume is True
    assert path == store_path_for(Path(".giskard/checkpoints"), fingerprint)


def test_resolve_checkpoint_dir_false_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GISKARD_CHECKPOINT", raising=False)
    path, resume = resolve_checkpoint_options(
        checkpoint_dir=False, fingerprint={"phase": "run"}
    )
    assert path is None
    assert resume is False


def test_resolve_env_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GISKARD_CHECKPOINT", "0")
    path, resume = resolve_checkpoint_options(fingerprint={"phase": "run"})
    assert path is None
    assert resume is False


def test_resolve_explicit_root_nests_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GISKARD_CHECKPOINT", raising=False)
    fingerprint = {"phase": "run", "suite": "x"}
    path, resume = resolve_checkpoint_options(
        checkpoint_dir=tmp_path / "ck", resume=False, fingerprint=fingerprint
    )
    assert resume is False
    assert path == store_path_for(tmp_path / "ck", fingerprint)


def test_resolve_api_resume_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GISKARD_CHECKPOINT", raising=False)
    monkeypatch.setenv("GISKARD_CHECKPOINT_RESUME", "force")
    _, resume = resolve_checkpoint_options(
        checkpoint_dir=False, resume=False, fingerprint={"phase": "run"}
    )
    assert resume is False
