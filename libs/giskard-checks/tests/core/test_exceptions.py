from giskard.checks import InputGenerationException


def test_input_generation_exception_is_exception():
    exc = InputGenerationException("schema issue: no string field")
    assert isinstance(exc, Exception)
    assert str(exc) == "schema issue: no string field"
