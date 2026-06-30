from giskard.checks import SuiteResult


class _FakeAdapter:
    async def run(self, target, **kwargs) -> SuiteResult:
        return SuiteResult(results=[], duration_ms=0)


def test_register_and_retrieve():
    from giskard.scan.integrations._registry import available, get, register

    register("_fake_test_tool", _FakeAdapter)
    assert get("_fake_test_tool") is _FakeAdapter
    assert "_fake_test_tool" in available()


def test_get_unknown_returns_none():
    from giskard.scan.integrations._registry import get

    assert get("no_such_tool_xyz_abc") is None
