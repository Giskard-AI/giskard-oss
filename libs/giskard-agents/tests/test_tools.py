"""Tests for the tools module."""

import json
from datetime import datetime, timezone
from typing import List
from uuid import UUID

import pytest
from giskard import agents
from giskard.agents.tools import Tool, tool
from pydantic import BaseModel


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


@pytest.mark.functional
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


# ---------------------------------------------------------------------------
# Tool.run() serialization — non-string return types
# ---------------------------------------------------------------------------


class CityWeather(BaseModel):
    city: str
    temp: float


@pytest.mark.parametrize(
    "return_value, expected",
    [
        pytest.param("sunny", "sunny", id="str-passthrough"),
        pytest.param({"temp": 22}, json.dumps({"temp": 22}), id="dict"),
        pytest.param([1, 2, 3], json.dumps([1, 2, 3]), id="list"),
        pytest.param(42, "42", id="int"),
        pytest.param(3.14, "3.14", id="float"),
        pytest.param(True, "true", id="bool"),
        pytest.param(None, "null", id="none"),
    ],
)
async def test_tool_run_serializes_to_str(return_value, expected):
    """Tool.run() returns str: strings as-is, everything else json.dumps'd."""

    @tool
    def stub(x: str) -> object:
        """Return a value.

        Parameters
        ----------
        x : str
            Ignored.
        """
        return return_value

    result = await stub.run({"x": "ignored"})
    assert result == expected
    assert isinstance(result, str)


async def test_tool_run_serializes_basemodel():
    """Tool.run() serializes BaseModel via TypeAdapter to JSON-safe str."""

    @tool
    def get_weather(city: str) -> CityWeather:
        """Get weather.

        Parameters
        ----------
        city : str
            City name.
        """
        return CityWeather(city=city, temp=22.5)

    result = await get_weather.run({"city": "Paris"})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == {"city": "Paris", "temp": 22.5}


# ---------------------------------------------------------------------------
# GAP-005: Input coercion
# ---------------------------------------------------------------------------


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address


class TimestampedRecord(BaseModel):
    id: UUID
    created_at: datetime
    label: str


async def test_tool_run_coerces_basemodel_input():
    """Tool.run() should coerce a dict into a BaseModel instance."""
    received = {}

    @tool
    def process_person(person: Person) -> str:
        """Process a person.

        Parameters
        ----------
        person : Person
            The person to process.
        """
        received["person"] = person
        return person.name

    result = await process_person.run(
        {"person": {"name": "Alice", "address": {"street": "123 Main", "city": "NYC"}}}
    )
    assert result == "Alice"
    assert isinstance(received["person"], Person)
    assert isinstance(received["person"].address, Address)


async def test_tool_run_coerces_optional_basemodel_input():
    """Tool.run() should coerce a dict into BaseModel | None."""
    received = {}

    @tool
    def process_optional(person: Person | None = None) -> str:
        """Process an optional person.

        Parameters
        ----------
        person : Person | None
            The person to process.
        """
        received["person"] = person
        return person.name if person else "nobody"

    result = await process_optional.run(
        {"person": {"name": "Bob", "address": {"street": "1 Elm", "city": "LA"}}}
    )
    assert result == "Bob"
    assert isinstance(received["person"], Person)

    result = await process_optional.run({"person": None})
    assert result == "nobody"
    assert received["person"] is None


async def test_tool_run_coerces_list_basemodel_input():
    """Tool.run() should coerce a list of dicts into list[BaseModel]."""
    received = {}

    @tool
    def process_people(people: list[Person]) -> int:
        """Process a list of people.

        Parameters
        ----------
        people : list[Person]
            People to process.
        """
        received["people"] = people
        return len(people)

    result = await process_people.run(
        {
            "people": [
                {"name": "A", "address": {"street": "1", "city": "X"}},
                {"name": "B", "address": {"street": "2", "city": "Y"}},
            ]
        }
    )
    assert result == "2"
    assert all(isinstance(p, Person) for p in received["people"])


# ---------------------------------------------------------------------------
# GAP-005: Output serialization via TypeAdapter
# ---------------------------------------------------------------------------


async def test_tool_run_serializes_basemodel_output_json_safe():
    """Tool.run() should produce JSON-safe str for BaseModel with rich types."""
    ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    uid = UUID("12345678-1234-5678-1234-567812345678")

    @tool
    def create_record(label: str) -> TimestampedRecord:
        """Create a record.

        Parameters
        ----------
        label : str
            The label.
        """
        return TimestampedRecord(id=uid, created_at=ts, label=label)

    result = await create_record.run({"label": "test"})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed["id"], str)
    assert isinstance(parsed["created_at"], str)
    assert parsed["label"] == "test"


async def test_tool_run_serializes_list_basemodel_output():
    """Tool.run() should serialize list[BaseModel] to JSON-safe str."""

    @tool
    def list_addresses(n: int) -> list[Address]:
        """List addresses.

        Parameters
        ----------
        n : int
            Count.
        """
        return [Address(street=f"St {i}", city=f"City {i}") for i in range(n)]

    result = await list_addresses.run({"n": 3})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert len(parsed) == 3
    assert all(isinstance(item, dict) for item in parsed)
    assert parsed[0] == {"street": "St 0", "city": "City 0"}


@pytest.mark.parametrize(
    "args, expected",
    [
        pytest.param({"query": "hello", "limit": 5}, "hello:5", id="str-int"),
        pytest.param({"query": "x", "limit": 0}, "x:0", id="str-zero"),
    ],
)
async def test_tool_run_primitive_types_unchanged(args, expected):
    """Primitive types should pass through coercion and serialization unchanged."""

    @tool
    def search(query: str, limit: int) -> str:
        """Search.

        Parameters
        ----------
        query : str
            Query text.
        limit : int
            Max results.
        """
        return f"{query}:{limit}"

    result = await search.run(args)
    assert result == expected
