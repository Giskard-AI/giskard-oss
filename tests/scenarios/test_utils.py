from datetime import datetime

import pytest
from giskard.checks.core import Interaction, Trace
from giskard.checks.scenarios.utils import execute_code, make_generator

VALUES = [
    1,
    "hello",
    True,
    False,
    None,
    [1, 2, 3],
    {"a": 1, "b": 2},
    datetime.now(),
]


class TestExecuteCode:
    @pytest.mark.parametrize("value", VALUES)
    async def test_execute_code_returns_static_value(self, value):
        result = await execute_code(value, Trace(interactions=[]))
        assert result == value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_callable_value_without_trace(
        self, return_value
    ):
        call_count = 0

        def callable():
            nonlocal call_count
            call_count += 1
            return return_value

        result = await execute_code(callable, Trace(interactions=[]))
        assert call_count == 1
        assert result == return_value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_awaitable_value_without_trace(
        self, return_value
    ):
        call_count = 0

        async def callable():
            nonlocal call_count
            call_count += 1
            return return_value

        result = await execute_code(callable, Trace(interactions=[]))
        assert call_count == 1
        assert result == return_value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_callable_value_with_trace(self, return_value):
        call_count = 0
        trace = Trace(
            interactions=[
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
        )

        def callable(trace: Trace[dict[str, str], dict[str, str]]):
            nonlocal call_count
            call_count += 1
            assert trace == Trace(
                interactions=[
                    Interaction(
                        inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                    )
                ]
            )
            return return_value

        result = await execute_code(callable, trace)
        assert call_count == 1
        assert result == return_value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_awaitable_value_with_trace(self, return_value):
        call_count = 0
        trace = Trace(
            interactions=[
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
        )

        async def callable(trace: Trace[dict[str, str], dict[str, str]]):
            nonlocal call_count
            call_count += 1
            assert trace == Trace(
                interactions=[
                    Interaction(
                        inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                    )
                ]
            )
            return return_value

        result = await execute_code(callable, trace)
        assert call_count == 1
        assert result == return_value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_callable_value_with_custom_trace(
        self, return_value
    ):
        call_count = 0
        trace = Trace(
            interactions=[
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
        )

        class CustomTrace(Trace[dict[str, str], dict[str, str]], frozen=True):
            def key(self) -> str:
                return self.interactions[0].inputs["key"]

        def callable(trace: CustomTrace):
            nonlocal call_count
            call_count += 1
            assert isinstance(trace, CustomTrace)
            assert trace.key() == "value"
            assert trace.interactions == [
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
            return return_value

        result = await execute_code(callable, trace)
        assert call_count == 1
        assert result == return_value

    @pytest.mark.parametrize("return_value", VALUES)
    async def test_execute_code_returns_awaitable_value_with_custom_trace(
        self, return_value
    ):
        call_count = 0
        trace = Trace(
            interactions=[
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
        )

        class CustomTrace(Trace[dict[str, str], dict[str, str]], frozen=True):
            def key(self) -> str:
                return self.interactions[0].inputs["key"]

        async def callable(trace: CustomTrace):
            nonlocal call_count
            call_count += 1
            assert isinstance(trace, CustomTrace)
            assert trace.key() == "value"
            assert trace.interactions == [
                Interaction(
                    inputs={"key": "value"}, outputs={"key": "value"}, metadata={}
                )
            ]
            return return_value

        result = await execute_code(callable, trace)
        assert call_count == 1
        assert result == return_value


class TestGenerate:
    async def test_generate_with_generator(self):
        trace = Trace(interactions=[])

        def generator():
            _ = yield 1
            _ = yield 2
            _ = yield 3

        gen = await make_generator(generator, trace)
        assert await anext(gen) == 1
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=1, metadata={})])
            )
            == 2
        )
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )
            == 3
        )
        with pytest.raises(StopAsyncIteration):
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=3, metadata={})])
            )

    async def test_generate_with_async_generator(self):
        trace = Trace(interactions=[])

        async def generator():
            _ = yield 1
            _ = yield 2
            _ = yield 3

        gen = await make_generator(generator, trace)
        assert await anext(gen) == 1
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=1, metadata={})])
            )
            == 2
        )
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )
            == 3
        )
        with pytest.raises(StopAsyncIteration):
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=3, metadata={})])
            )

    async def test_generate_with_generator_and_custom_trace(self):
        trace = Trace(interactions=[])

        class CustomTrace(Trace[int, int], frozen=True):
            def outputs(self) -> int:
                return self.interactions[-1].outputs

        def generator(trace: CustomTrace):
            assert isinstance(trace, CustomTrace)
            trace = yield 1
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 1
            trace = yield 2
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 2
            trace = yield 3
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 3

        gen = await make_generator(generator, trace)
        assert await anext(gen) == 1
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=1, metadata={})])
            )
            == 2
        )
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )
            == 3
        )
        with pytest.raises(StopAsyncIteration):
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=3, metadata={})])
            )

    async def test_generate_with_async_generator_and_custom_trace(self):
        trace = Trace(interactions=[])

        class CustomTrace(Trace[int, int], frozen=True):
            def outputs(self) -> int:
                return self.interactions[-1].outputs

        async def generator(trace: CustomTrace):
            assert isinstance(trace, CustomTrace)
            trace = yield 1
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 1
            trace = yield 2
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 2
            trace = yield 3
            assert isinstance(trace, CustomTrace)
            assert trace.outputs() == 3

        gen = await make_generator(generator, trace)
        assert await anext(gen) == 1
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=1, metadata={})])
            )
            == 2
        )
        assert (
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=2, metadata={})])
            )
            == 3
        )
        with pytest.raises(StopAsyncIteration):
            await gen.asend(
                Trace(interactions=[Interaction(inputs=1, outputs=3, metadata={})])
            )
