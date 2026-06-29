"""Tests for the RegoPolicy check."""

import importlib
import importlib.util
import json
from typing import Any

import pytest
from giskard.checks import Check, CheckStatus, Interaction, RegoPolicy, Trace
from giskard.checks.utils import optional_deps
from pydantic import ValidationError

BOOLEAN_POLICY = """\
package giskard

default allow = false

allow if {
    input.role == "admin"
}
"""

INVALID_POLICY = """\
package giskard

default allow = false

allow if {
    input.role === "admin"
}
"""

_regorus_installed = importlib.util.find_spec("regorus") is not None
_skip_if_no_regorus = pytest.mark.skipif(
    not _regorus_installed,
    reason="regorus not installed (pip install 'giskard-checks[regorus]')",
)


def _check(**kwargs: Any) -> RegoPolicy:  # pyright: ignore[reportMissingTypeArgument]
    defaults: dict[str, Any] = {
        "policy": BOOLEAN_POLICY,
        "rule": "data.giskard.allow",
    }
    defaults.update(kwargs)
    return RegoPolicy(**defaults)


@pytest.fixture
def block_regorus_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, /, *args, **kwargs):
        if name == "regorus":
            raise ImportError("missing regorus")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(optional_deps.importlib, "import_module", fake_import_module)


def test_rego_policy_requires_regorus_on_instantiation(
    block_regorus_import: None,
) -> None:
    with pytest.raises(ValidationError, match="giskard-checks\\[regorus\\]"):
        _check()


def test_rego_policy_requires_regorus_on_model_validate_json(
    block_regorus_import: None,
) -> None:
    payload = json.dumps(
        {
            "kind": "rego_policy",
            "policy": BOOLEAN_POLICY,
            "rule": "data.giskard.allow",
        }
    )
    with pytest.raises(ValidationError, match="giskard-checks\\[regorus\\]"):
        Check.model_validate_json(payload)


async def test_no_match_on_key_fails() -> None:
    check = _check(key="trace.last.metadata.missing")
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "admin"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.message is not None
    assert "No value found" in result.message


async def test_serialization_round_trip() -> None:
    check = _check(name="authz", data={"roles": ["admin"]})
    payload = check.model_dump()
    restored = Check.model_validate(payload)

    assert isinstance(restored, RegoPolicy)
    assert restored.policy == BOOLEAN_POLICY
    assert restored.rule == "data.giskard.allow"
    assert restored.data == {"roles": ["admin"]}


@_skip_if_no_regorus
async def test_allow_passes() -> None:
    check = _check()
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "admin"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.passed
    assert result.details["value"] is True


@_skip_if_no_regorus
async def test_deny_fails() -> None:
    check = _check()
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "user"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.details["value"] is False


@_skip_if_no_regorus
async def test_undefined_rule_fails() -> None:
    policy = """\
package giskard

allow if {
    input.role == "admin"
}
"""
    check = _check(policy=policy)
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "user"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert result.details["value"] is None
    assert result.message is not None
    assert "undefined" in result.message


@_skip_if_no_regorus
async def test_non_boolean_rule_returns_error() -> None:
    policy = """\
package giskard

message := "denied" if {
    true
}
"""
    check = _check(policy=policy, rule="data.giskard.message")
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "admin"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert result.details["value"] == "denied"
    assert result.message is not None
    assert "expected bool" in result.message


@_skip_if_no_regorus
@pytest.mark.parametrize(
    ("key", "outputs", "policy", "expected_status", "expected_input"),
    [
        pytest.param(
            "trace.last.outputs.payload",
            {"payload": {"role": "admin"}},
            """\
package giskard

default allow = false

allow if {
    input.role == "admin"
}
""",
            CheckStatus.PASS,
            {"role": "admin"},
            id="dict",
        ),
        pytest.param(
            "trace.last.outputs.items",
            {"items": [1, 2, 3]},
            """\
package giskard

default allow = false

allow if {
    input[0] == 1
    count(input) == 3
}
""",
            CheckStatus.PASS,
            [1, 2, 3],
            id="list",
        ),
        pytest.param(
            "trace.last.outputs.label",
            {"label": "approved"},
            """\
package giskard

default allow = false

allow if {
    input == "approved"
}
""",
            CheckStatus.PASS,
            "approved",
            id="string",
        ),
        pytest.param(
            "trace.last.outputs.count",
            {"count": 42},
            """\
package giskard

default allow = false

allow if {
    input == 42
}
""",
            CheckStatus.PASS,
            42,
            id="number",
        ),
        pytest.param(
            "trace.last.outputs.active",
            {"active": True},
            """\
package giskard

default allow = false

allow if {
    input == true
}
""",
            CheckStatus.PASS,
            True,
            id="boolean",
        ),
        pytest.param(
            "trace.last.outputs.empty",
            {"empty": None},
            """\
package giskard

default allow = false

allow if {
    input == null
}
""",
            CheckStatus.PASS,
            None,
            id="null",
        ),
    ],
)
async def test_json_serializable_input_types(
    key: str,
    outputs: dict[str, Any],
    policy: str,
    expected_status: CheckStatus,
    expected_input: Any,
) -> None:
    """JSONPath can target any JSON-serializable value passed via set_input_json."""
    check = _check(policy=policy, key=key)
    trace = await Trace.from_interactions(Interaction(inputs="test", outputs=outputs))

    result = await check.run(trace)

    assert result.status == expected_status
    assert result.details["input"] == expected_input


@_skip_if_no_regorus
async def test_data_document_used() -> None:
    policy = """\
package giskard

default allow = false

allow if {
    input.role in data.allowed_roles
}
"""
    check = _check(
        policy=policy,
        data={"allowed_roles": ["admin", "editor"]},
    )
    trace = await Trace.from_interactions(
        Interaction(inputs="test", outputs={"role": "editor"})
    )

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS


@_skip_if_no_regorus
async def test_invalid_policy_returns_error() -> None:
    with pytest.raises(ValidationError, match="Invalid Rego policy or data"):
        _check(policy=INVALID_POLICY)
