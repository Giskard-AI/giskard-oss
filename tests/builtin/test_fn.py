import pytest
from giskard.checks import (
    CheckResult,
    CheckStatus,
    FnCheck,
    Interaction,
    Trace,
    from_fn,
)


async def test_sync_function_returns_true() -> None:
    """Test FnCheck with sync function returning True."""
    check = FnCheck(fn=lambda trace: True)
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.passed
    assert not result.failed


async def test_sync_function_returns_false() -> None:
    """Test FnCheck with sync function returning False."""
    check = FnCheck(fn=lambda trace: False)
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.failed
    assert not result.passed


async def test_async_function_returns_true() -> None:
    """Test FnCheck with async function returning True."""

    async def async_fn(trace: Trace[dict[str, str], dict[str, str]]) -> bool:
        return True

    check = FnCheck(fn=async_fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.passed


async def test_async_function_returns_false() -> None:
    """Test FnCheck with async function returning False."""

    async def async_fn(trace: Trace[dict[str, str], dict[str, str]]) -> bool:
        return False

    check = FnCheck(fn=async_fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.failed


async def test_sync_function_returns_check_result() -> None:
    """Test FnCheck with sync function returning CheckResult."""

    def fn(trace: Trace[dict[str, str], dict[str, str]]) -> CheckResult:
        return CheckResult.success(message="Custom success message")

    check = FnCheck(fn=fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message == "Custom success message"
    assert result.passed


async def test_async_function_returns_check_result() -> None:
    """Test FnCheck with async function returning CheckResult."""

    async def async_fn(trace: Trace[dict[str, str], dict[str, str]]) -> CheckResult:
        return CheckResult.failure(message="Custom failure message")

    check = FnCheck(fn=async_fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message == "Custom failure message"
    assert result.failed


async def test_success_message() -> None:
    """Test FnCheck with success_message set."""
    check = FnCheck(
        fn=lambda trace: True,
        success_message="Check passed successfully",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.message == "Check passed successfully"


async def test_failure_message() -> None:
    """Test FnCheck with failure_message set."""
    check = FnCheck(
        fn=lambda trace: False,
        failure_message="Check failed",
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.FAIL
    assert result.message == "Check failed"


async def test_details() -> None:
    """Test FnCheck with details set."""
    check = FnCheck(
        fn=lambda trace: True,
        details={"key": "value", "number": 42},
    )
    result = await check.run(Trace())
    assert result.status == CheckStatus.PASS
    assert result.details["key"] == "value"
    assert result.details["number"] == 42


async def test_function_receives_trace() -> None:
    """Test that the function receives the trace correctly."""
    received_trace: Trace[dict[str, str], dict[str, str]] | None = None

    def fn(trace: Trace[dict[str, str], dict[str, str]]) -> bool:
        nonlocal received_trace
        received_trace = trace
        return True

    check = FnCheck(fn=fn)
    test_trace = Trace(
        interactions=[
            Interaction(inputs={"query": "test"}, outputs={"response": "answer"})
        ]
    )
    result = await check.run(test_trace)
    assert result.passed
    assert received_trace is not None
    assert len(received_trace.interactions) == 1
    assert received_trace.interactions[0].inputs["query"] == "test"


async def test_from_fn_basic() -> None:
    """Test from_fn factory function with basic usage."""
    check = from_fn(lambda trace: True, name="test_check")
    result = await check.run(Trace())
    assert result.passed
    assert check.name == "test_check"


async def test_from_fn_with_description() -> None:
    """Test from_fn with description."""
    check = from_fn(
        lambda trace: True,
        name="test_check",
        description="A test check",
    )
    result = await check.run(Trace())
    assert result.passed
    assert check.description == "A test check"


async def test_from_fn_with_messages() -> None:
    """Test from_fn with success and failure messages."""
    check = from_fn(
        lambda trace: True,
        success_message="Success!",
        failure_message="Failed!",
    )
    result = await check.run(Trace())
    assert result.passed
    assert result.message == "Success!"

    check_fail = from_fn(
        lambda trace: False,
        success_message="Success!",
        failure_message="Failed!",
    )
    result_fail = await check_fail.run(Trace())
    assert result_fail.failed
    assert result_fail.message == "Failed!"


async def test_from_fn_with_details() -> None:
    """Test from_fn with details."""
    check = from_fn(
        lambda trace: True,
        details={"custom": "data"},
    )
    result = await check.run(Trace())
    assert result.passed
    assert result.details["custom"] == "data"


async def test_check_result_overrides_messages() -> None:
    """Test that CheckResult returned from function overrides success/failure messages."""

    def fn(trace: Trace[str, str]) -> CheckResult:
        return CheckResult.success(message="From function")

    check = FnCheck(
        fn=fn,
        success_message="From check",
    )
    result = await check.run(Trace())
    # CheckResult message takes precedence
    assert result.message == "From function"


async def test_check_result_with_details() -> None:
    """Test CheckResult with details from function."""

    def fn(trace: Trace[str, str]) -> CheckResult:
        return CheckResult.success(
            message="Success",
            details={"from_function": True},
        )

    check = FnCheck(
        fn=fn,
        details={"from_check": True},
    )
    result = await check.run(Trace())
    # CheckResult details take precedence
    assert result.details["from_function"] is True
    assert "from_check" not in result.details


async def test_invalid_return_type() -> None:
    """Test that invalid return type raises TypeError."""
    check = FnCheck(fn=lambda _trace: "invalid")  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="must return bool or CheckResult"):
        _ = await check.run(Trace())


async def test_invalid_async_return_type() -> None:
    """Test that invalid async return type raises TypeError."""

    async def invalid_fn(trace: Trace[dict[str, str], dict[str, str]]) -> str:
        return "invalid"

    check = FnCheck(fn=invalid_fn)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="must return bool or CheckResult"):
        await check.run(Trace())


async def test_complex_trace_usage() -> None:
    """Test function that uses trace data to make decision."""

    def check_has_outputs(trace: Trace[dict[str, str], dict[str, str] | None]) -> bool:
        if not trace.interactions:
            return False
        last_interaction = trace.interactions[-1]
        return last_interaction.outputs is not None

    check = FnCheck(fn=check_has_outputs)

    # Test with trace that has outputs
    trace_with_outputs = Trace[dict[str, str], dict[str, str] | None](
        interactions=[Interaction(inputs={"q": "test"}, outputs={"a": "answer"})]
    )
    result = await check.run(trace_with_outputs)
    assert result.passed

    # Test with trace that has no outputs
    trace_no_outputs = Trace[dict[str, str], dict[str, str] | None](
        interactions=[Interaction(inputs={"q": "test"}, outputs=None)]
    )
    result = await check.run(trace_no_outputs)
    assert result.failed

    # Test with empty trace
    result = await check.run(Trace())
    assert result.failed


async def test_check_result_error_status() -> None:
    """Test function returning CheckResult with ERROR status."""

    def fn(trace: Trace[str, str]) -> CheckResult:
        return CheckResult.error(message="An error occurred")

    check = FnCheck(fn=fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.ERROR
    assert result.errored
    assert result.message == "An error occurred"


async def test_check_result_skip_status() -> None:
    """Test function returning CheckResult with SKIP status."""

    def fn(trace: Trace[str, str]) -> CheckResult:
        return CheckResult.skip(message="Skipped due to precondition")

    check = FnCheck(fn=fn)
    result = await check.run(Trace())
    assert result.status == CheckStatus.SKIP
    assert result.skipped
    assert result.message == "Skipped due to precondition"
