"""Suite.run checkpoint / resume integration tests."""

import json
from pathlib import Path
from typing import Any

from giskard.checks import Equals, RunStore, Scenario, Suite
from giskard.checks.utils.checkpoint import run_fingerprint, store_path_for


def _echo(inputs: str) -> str:
    return inputs


def _suite_store_path(root: Path, suite: Suite[Any, Any]) -> Path:
    from giskard.checks.utils.checkpoint import ensure_checkpoint_id

    ids = [
        ensure_checkpoint_id(
            scenario.annotations,
            name=scenario.name,
            index=index,
            suite_name=suite.name,
        )
        for index, scenario in enumerate(suite.scenarios)
    ]
    return store_path_for(root, run_fingerprint(suite.name, ids))


async def test_suite_run_writes_checkpoint_per_scenario(tmp_path: Path) -> None:
    suite = Suite(name="demo", target=_echo)
    suite.append(
        Scenario("a")
        .interact("a")
        .check(Equals(expected_value="a", key="trace.last.outputs"))
    )
    suite.append(
        Scenario("b")
        .interact("b")
        .check(Equals(expected_value="b", key="trace.last.outputs"))
    )

    result = await suite.run(checkpoint_dir=tmp_path / "ck", verbose=False)
    assert result.passed_count == 2

    events_path = _suite_store_path(tmp_path / "ck", suite) / "events.jsonl"
    lines = events_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["type"] == "scenario_finished"
        assert "id" in record
        assert "payload" in record


async def test_suite_run_default_resume_skips_completed(tmp_path: Path) -> None:
    calls: list[str] = []

    def tracking_target(inputs: str) -> str:
        calls.append(inputs)
        return inputs

    suite = Suite(name="demo", target=tracking_target)
    suite.append(
        Scenario("a")
        .interact("a")
        .check(Equals(expected_value="a", key="trace.last.outputs"))
    )
    suite.append(
        Scenario("b")
        .interact("b")
        .check(Equals(expected_value="b", key="trace.last.outputs"))
    )

    ck = tmp_path / "ck"
    first = await suite.run(checkpoint_dir=ck, verbose=False)
    assert first.passed_count == 2
    assert calls == ["a", "b"]

    calls.clear()
    # resume defaults to True
    second = await suite.run(checkpoint_dir=ck, verbose=False)
    assert second.passed_count == 2
    assert calls == []
    assert [r.scenario_name for r in second.results] == ["a", "b"]


async def test_suite_run_parallel_resume_preserves_order(tmp_path: Path) -> None:
    suite = Suite(name="demo", target=_echo)
    for name in ("a", "b", "c"):
        suite.append(
            Scenario(name)
            .interact(name)
            .check(Equals(expected_value=name, key="trace.last.outputs"))
        )

    ck = tmp_path / "ck"
    await suite.run(checkpoint_dir=ck, parallel=True, verbose=False)
    store_path = _suite_store_path(ck, suite)

    events = (store_path / "events.jsonl").read_text().strip().splitlines()
    assert len(events) == 3
    ids = [json.loads(line)["id"] for line in events]
    payloads = [json.loads(line)["payload"] for line in events]
    partial = tmp_path / "partial"
    fingerprint = json.loads((store_path / "manifest.json").read_text())["fingerprint"]
    partial_store = await RunStore.open(
        store_path_for(partial, fingerprint),
        fingerprint=fingerprint,
        resume=False,
    )
    await partial_store.append("scenario_finished", id=ids[0], payload=payloads[0])
    await partial_store.append("scenario_finished", id=ids[1], payload=payloads[1])

    calls: list[str] = []

    def tracking(inputs: str) -> str:
        calls.append(inputs)
        return inputs

    suite2 = Suite(name="demo", target=tracking)
    for name in ("a", "b", "c"):
        suite2.append(
            Scenario(name)
            .interact(name)
            .check(Equals(expected_value=name, key="trace.last.outputs"))
        )

    result = await suite2.run(checkpoint_dir=partial, parallel=True, verbose=False)
    assert [r.scenario_name for r in result.results] == ["a", "b", "c"]
    assert calls == ["c"]


async def test_suite_run_checkpoint_dir_false_writes_nothing(tmp_path: Path) -> None:
    suite = Suite(name="demo", target=_echo)
    suite.append(
        Scenario("a")
        .interact("a")
        .check(Equals(expected_value="a", key="trace.last.outputs"))
    )
    await suite.run(checkpoint_dir=False, verbose=False)
    assert list(tmp_path.iterdir()) == []
