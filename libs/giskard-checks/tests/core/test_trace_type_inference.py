from giskard.checks.core.interaction.trace import Trace
from giskard.checks.utils.inference import _infer_trace_type


class MyTrace(Trace[str, str], frozen=True):
    pass


class OtherTrace(Trace[str, str], frozen=True):
    pass


def test_infer_trace_type_returns_none_for_non_callable():
    assert _infer_trace_type("not a callable") is None


def test_infer_trace_type_returns_none_for_single_param_callable():
    def target(inputs: str) -> str:
        return inputs

    assert _infer_trace_type(target) is None


def test_infer_trace_type_returns_none_when_second_param_not_trace():
    def target(inputs: str, context: dict[str, object]) -> str:
        return inputs

    assert _infer_trace_type(target) is None


def test_infer_trace_type_returns_subclass_when_second_param_is_trace_subclass():
    def target(inputs: str, trace: MyTrace) -> str:
        return inputs

    assert _infer_trace_type(target) is MyTrace


def test_infer_trace_type_returns_base_trace_when_second_param_is_base_trace():
    def target(inputs: str, trace: Trace[str, str]) -> str:
        return inputs

    result = _infer_trace_type(target)
    assert result is Trace[str, str]


def test_infer_trace_type_works_for_callable_instance():
    class MyAgent:
        def __call__(self, inputs: str, trace: MyTrace) -> str:
            return inputs

    assert _infer_trace_type(MyAgent()) is MyTrace


def test_infer_trace_type_returns_none_for_callable_instance_no_second_param():
    class MyAgent:
        def __call__(self, inputs: str) -> str:
            return inputs

    assert _infer_trace_type(MyAgent()) is None


def test_infer_trace_type_returns_none_for_callable_with_no_annotations():
    assert _infer_trace_type(lambda x, y: x) is None
