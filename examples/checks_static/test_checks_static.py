"""Offline checks happy path — mirrors the canonical example contract."""

import asyncio

from giskard.checks import CheckResult, Equals, Scenario, from_fn


def echo(inputs: str) -> str:
    return inputs


@from_fn
async def not_empty(trace) -> CheckResult:
    outputs = trace.last.outputs
    if str(outputs).strip():
        return CheckResult.success(message="non-empty")
    return CheckResult.failure(message="empty")


async def main() -> None:
    result = await (
        Scenario("echo")
        .interact(inputs="hello", outputs=echo)
        .check(Equals(key="trace.last.outputs", expected_value="hello"))
        .check(not_empty)
        .run()
    )
    assert result.passed
    result.print_report()


async def test_checks_static_happy_path() -> None:
    await main()


if __name__ == "__main__":
    asyncio.run(main())
