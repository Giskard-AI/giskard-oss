"""Tests for the tools module."""

from typing import List

import pytest
from giskard import agents
from giskard.agents.tools import Tool, tool


def test_tool_decorator():
    """Test that the tool decorator works correctly."""

    @tool
    def search_web(query: str, max_results: int = 10) -> List[str]:
        """Retrieve search results from the web.

        This is a test tool.

        Parameters
        ----------
        query : str
            The search query to use.
        max_results : int, optional
            Maximum number of documents that should be returned for the call.
        """
        return ["This is a test", f"another test for {query}"]

    assert isinstance(search_web, Tool)

    assert search_web.name == "search_web"
    assert (
        search_web.description
        == "Retrieve search results from the web.\n\nThis is a test tool."
    )

    # Check schema
    assert search_web.parameters_schema["type"] == "object"
    assert list(search_web.parameters_schema["properties"].keys()) == [
        "query",
        "max_results",
    ]

    assert search_web.parameters_schema["properties"]["query"]["type"] == "string"
    assert (
        search_web.parameters_schema["properties"]["query"]["description"]
        == "The search query to use."
    )

    assert (
        search_web.parameters_schema["properties"]["max_results"]["type"] == "integer"
    )
    assert (
        search_web.parameters_schema["properties"]["max_results"]["description"]
        == "Maximum number of documents that should be returned for the call."
    )

    assert search_web.parameters_schema["required"] == ["query"]

    assert search_web("Q", max_results=5) == ["This is a test", "another test for Q"]


async def test_tool_with_methods():
    """Test that the tool runs correctly."""

    class Weather:
        @tool
        async def get_weather(self, city: str) -> str:
            """Get the weather in a city."""
            return f"It's sunny in {city}."

    weather = Weather()

    assert isinstance(weather.get_weather, Tool)
    assert await weather.get_weather.run({"city": "Paris"}) == "It's sunny in Paris."

    # Test calling the tool directly like a regular method
    assert await weather.get_weather("London") == "It's sunny in London."

    # Test calling with keyword arguments
    assert await weather.get_weather(city="Tokyo") == "It's sunny in Tokyo."


async def test_tool_run(generator):
    """Test that the tool runs correctly."""

    @agents.tool
    def get_weather(city: str) -> str:
        """Get the weather in a city.

        Parameters
        ----------
        city: str
            The city to get the weather for.
        """
        if city == "Paris":
            return f"It's raining in {city}."

        return f"It's sunny in {city}."

    chat = await (
        generator.chat("Hello, what's the weather in Paris?")
        .with_tools(get_weather)
        .run()
    )

    assert "rain" in chat.last.content.lower()


async def test_tool_catches_errors(generator):
    """Test that the tool catches errors correctly."""

    @agents.tool
    def get_weather(city: str) -> str:
        raise ValueError("City not found")

    result = await get_weather.run({"city": "Paris"})
    assert result == "ERROR: City not found"

    # The original function behavior is not modified
    with pytest.raises(ValueError):
        get_weather("Paris")


async def test_tool_catches_errors_with_async_function(generator):
    """Test that the tool catches errors correctly."""

    @agents.tool
    async def get_weather(city: str) -> str:
        raise ValueError("City not found")

    result = await get_weather.run({"city": "Paris"})
    assert result == "ERROR: City not found"

    # The original function behavior is not modified
    with pytest.raises(ValueError):
        await get_weather("Paris")


async def test_tool_does_not_catch_errors(generator):
    """Test that the tool catches errors correctly."""

    @agents.tool(catch=None)
    def get_weather(city: str) -> str:
        raise ValueError("City not found")

    with pytest.raises(ValueError):
        await get_weather.run({"city": "Paris"})

    # The original function behavior is not modified
    with pytest.raises(ValueError):
        get_weather("Paris")


async def test_tool_does_not_catch_errors_with_async_function(generator):
    """Test that the tool catches errors correctly."""

    @agents.tool(catch=None)
    async def get_weather(city: str) -> str:
        raise ValueError("City not found")

    with pytest.raises(ValueError):
        await get_weather.run({"city": "Paris"})

    # The original function behavior is not modified
    with pytest.raises(ValueError):
        await get_weather("Paris")


async def test_tool_method_catches_errors(generator):
    """Test that the tool method catches errors correctly."""

    class Weather:
        @agents.tool
        def get_weather(self, city: str) -> str:
            raise ValueError("City not found")

    weather = Weather()
    result = await weather.get_weather.run({"city": "Paris"})
    assert result == "ERROR: City not found"

    # The original function behavior is not modified
    with pytest.raises(ValueError):
        weather.get_weather("Paris")
