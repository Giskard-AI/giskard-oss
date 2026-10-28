"""PEP 561 marker is packaged with giskard.llm."""

from importlib.resources import files


def test_pep561_py_typed_marker_is_present():
    marker = files("giskard.llm").joinpath("py.typed")
    assert marker.is_file()
