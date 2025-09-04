from __future__ import annotations

import base64
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from giskard_checks.testing.testcase import TestCase


def _interaction_module() -> str:
    return "giskard_checks.testing._samples.custom_interaction"


def _checks_module() -> str:
    return "giskard_checks.testing._samples.custom_checks"


async def test_serialize_then_deserialize_after_import():
    # Build a payload in-process
    inter_mod = importlib.import_module(_interaction_module())
    CustomInteraction = getattr(inter_mod, "CustomInteraction")

    checks_mod = importlib.import_module(_checks_module())
    StartsWithCheck = getattr(checks_mod, "StartsWithCheck")
    EqualsOutputCheck = getattr(checks_mod, "EqualsOutputCheck")

    interaction = CustomInteraction(input="hello world", output="ok")
    chk1 = StartsWithCheck(name="starts_hello", prefix="hello")
    chk2 = EqualsOutputCheck(name="out_ok", expected="ok")

    tc = TestCase(name="tc-unloaded", interaction=interaction, checks=[chk1, chk2])
    payload = tc.serialize()

    # Prepare payload for subprocess via base64 JSON
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    env = {
        **os.environ,
        "PYTHONPATH": repo_src + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    # 1) In a fresh interpreter without importing checks, deserialization should now succeed
    script_fail = f"""
import base64, json, sys
from giskard_checks.testing.testcase import TestCase
data = json.loads(base64.b64decode({payload_b64!r}).decode())
tc = TestCase.deserialize(data)
assert [c.kind for c in tc.checks] == [\"starts_with\", \"equals_out\"]
"""
    proc1 = subprocess.run([sys.executable, "-c", script_fail], env=env)
    assert proc1.returncode == 0

    # 2) And with explicit import it should also succeed (idempotent)
    script_ok = f"""
import base64, json, sys
from giskard_checks.testing.testcase import TestCase
import giskard_checks.testing._samples.custom_checks  # registers kinds
data = json.loads(base64.b64decode({payload_b64!r}).decode())
tc = TestCase.deserialize(data)
assert [c.kind for c in tc.checks] == ["starts_with", "equals_out"]
"""
    proc2 = subprocess.run([sys.executable, "-c", script_ok], env=env)
    assert proc2.returncode == 0

    # As a final sanity check: in-process deserialization still works and runs
    tc2 = TestCase.deserialize(payload)
    from giskard_checks.core.check import CheckStatus

    result = await tc2.run()
    statuses = [r.status for r in result.results]
    assert statuses == [CheckStatus.PASS, CheckStatus.PASS]
