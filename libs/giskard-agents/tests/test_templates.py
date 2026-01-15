import tempfile
from pathlib import Path

import pytest
from giskard.agents.templates import LLMFormattable, MessageTemplate, PromptsManager
from pydantic import BaseModel


@pytest.fixture
def prompts_manager():
    return PromptsManager(
        default_prompts_path=Path(__file__).parent / "data" / "prompts"
    )


async def test_message_template():
    template = MessageTemplate(
        role="user",
        content_template="Hello, {{ name }}!",
    )

    message = template.render(name="Orlande de Lassus")

    assert message.role == "user"
    assert message.content == "Hello, Orlande de Lassus!"


async def test_multi_message_template_parsing(prompts_manager):
    messages = await prompts_manager.render_template(
        "multi_message.j2",
        {
            "theory": "Normandy is actually the center of the universe because its perfect balance of rain, cheese, and cider creates a quantum field that bends space-time."
        },
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert (
        "You are an impartial evaluator of scientific theories" in messages[0].content
    )


async def test_invalid_template(prompts_manager):
    with pytest.raises(ValueError):
        await prompts_manager.render_template("invalid.j2")


async def test_simple_template(prompts_manager):
    messages = await prompts_manager.render_template("simple.j2")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert (
        messages[0].content
        == "This is a simple prompt that should be rendered as a single user message."
    )


def test_pydantic_json_rendering_inline():
    class Book(BaseModel):
        title: str
        description: str

    template = MessageTemplate(
        role="user",
        content_template="Hello, consider this content:\n{{ book }}!",
    )

    book = Book(
        title="The Great Gatsby",
        description="The Great Gatsby is a novel by F. Scott Fitzgerald.",
    )

    message = template.render(book=book)

    assert message.role == "user"
    expected_json = """{
    "title": "The Great Gatsby",
    "description": "The Great Gatsby is a novel by F. Scott Fitzgerald."
}"""
    assert message.content == f"Hello, consider this content:\n{expected_json}!"


async def test_pydantic_json_rendering_with_prompts_manager():
    class Book(BaseModel):
        title: str
        description: str

    book = Book(
        title="The Great Gatsby",
        description="The Great Gatsby is a novel by F. Scott Fitzgerald.",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_manager = PromptsManager(default_prompts_path=tmp_dir)

        template_path = Path(tmp_dir) / "book.j2"
        template_path.write_text("Here is a book:\n{{ book }}")

        messages = await prompts_manager.render_template("book.j2", {"book": book})

        assert len(messages) == 1
        assert messages[0].role == "user"
        expected_json = """{
    "title": "The Great Gatsby",
    "description": "The Great Gatsby is a novel by F. Scott Fitzgerald."
}"""
        assert messages[0].content == f"Here is a book:\n{expected_json}"


def test_llm_formattable_protocol_explicit_implementation():
    """Test that a class explicitly implementing the protocol is formatted correctly."""

    class CustomFormatter(LLMFormattable):
        def __init__(self, value: str):
            self.value = value

        def _repr_prompt_(self) -> str:
            return f"Formatted: {self.value}"

    # Verify it matches the protocol
    formatter = CustomFormatter("test")
    assert isinstance(formatter, LLMFormattable)

    template = MessageTemplate(
        role="user",
        content_template="Content: {{ obj }}",
    )

    message = template.render(obj=formatter)
    assert message.content == "Content: Formatted: test"


def test_llm_formattable_protocol_structural_typing():
    """Test that a class with the method (without explicit protocol) is formatted correctly."""

    class DuckTypedFormatter:
        """Class that implements the protocol method without explicitly mentioning it."""

        def __init__(self, data: str):
            self.data = data

        def _repr_prompt_(self) -> str:
            return f"Duck-typed: {self.data}"

    # Verify it matches the protocol (structural typing)
    formatter = DuckTypedFormatter("example")
    assert isinstance(formatter, LLMFormattable)

    template = MessageTemplate(
        role="user",
        content_template="Result: {{ obj }}",
    )

    message = template.render(obj=formatter)
    assert message.content == "Result: Duck-typed: example"


def test_llm_formattable_method_with_params_raises_type_error():
    """Test that a method requiring params raises TypeError when called.

    Note: This is a Python limitation - @runtime_checkable only checks method
    existence, not signatures. So isinstance() will pass, but calling the method
    will raise TypeError.
    """

    class WrongSignature:
        """Class with _repr_prompt_ that requires params - matches but fails when called."""

        def __init__(self, value: str):
            self.value = value
            self.called = False

        def _repr_prompt_(self, param: str) -> str:  # Wrong signature - requires param
            self.called = True
            return f"Should not be called: {self.value}"

    obj = WrongSignature("test")
    # @runtime_checkable limitation: only checks method existence, not signature
    # So this will pass isinstance() even though signature is wrong
    assert isinstance(obj, LLMFormattable)  # This is the Python limitation

    template = MessageTemplate(
        role="user",
        content_template="Value: {{ obj }}",
    )

    # Should raise TypeError when trying to call method with wrong signature
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        template.render(obj=obj)
    assert not obj.called  # Method shouldn't be successfully called


def test_llm_formattable_pydantic_with_wrong_signature_raises_type_error():
    """Test that a Pydantic model with _repr_prompt_ (wrong signature) raises TypeError."""

    class FormattablePydanticWrongSignature(BaseModel):
        """A Pydantic class with _repr_prompt_ that requires params - should raise TypeError."""

        value: str
        number: int

        def _repr_prompt_(self, param: str) -> str:  # Wrong signature - requires param
            return f"Should not be called: {self.value}"

    obj = FormattablePydanticWrongSignature(value="test", number=42)
    # @runtime_checkable limitation: only checks method existence, not signature
    # So this will pass isinstance() even though signature is wrong
    assert isinstance(obj, LLMFormattable)  # This is the Python limitation
    assert isinstance(obj, BaseModel)

    template = MessageTemplate(
        role="user",
        content_template="Content: {{ obj }}",
    )

    # Should raise TypeError when trying to call method with wrong signature
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        template.render(obj=obj)


def test_llm_formattable_method_returns_non_string_still_called():
    """Test that a method returning non-string still matches and is called (return type not checked at runtime)."""

    class WrongReturnType:
        """Class with _repr_prompt_ that returns non-string - matches protocol but returns wrong type."""

        def __init__(self, value: str):
            self.value = value
            self.called = False

        def _repr_prompt_(
            self,
        ) -> int:  # Wrong return type annotation, but runtime doesn't check
            self.called = True
            return 42  # Returns int, not str

    obj = WrongReturnType("test")
    # @runtime_checkable only checks method existence and signature (params), not return type
    # So this will match the protocol
    assert isinstance(obj, LLMFormattable)

    template = MessageTemplate(
        role="user",
        content_template="Value: {{ obj }}",
    )

    # The method will be called and return 42, which will be converted to string in template
    message = template.render(obj=obj)
    assert obj.called
    # The integer 42 will be converted to string "42" in the template
    assert message.content == "Value: 42"


def test_llm_formattable_takes_precedence_over_pydantic():
    """Test that LLMFormattable protocol takes precedence over Pydantic BaseModel."""

    class FormattablePydantic(BaseModel):
        """A Pydantic class that also implements the protocol method."""

        value: str

        def _repr_prompt_(self) -> str:
            return f"Protocol format: {self.value}"

    obj = FormattablePydantic(value="test")
    # Should match the protocol due to structural typing
    assert isinstance(obj, LLMFormattable)
    assert isinstance(obj, BaseModel)

    template = MessageTemplate(
        role="user",
        content_template="Content: {{ obj }}",
    )

    message = template.render(obj=obj)
    # Should use protocol method, not Pydantic JSON dump
    assert message.content == "Content: Protocol format: test"
    assert "Protocol format" in message.content
    assert '"value"' not in message.content  # Should not be JSON
