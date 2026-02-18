import json
from importlib.util import find_spec
from unittest.mock import MagicMock, Mock, patch

import pydantic
import pytest
from google.generativeai.types import ContentDict
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from giskard.llm.client import ChatMessage
from giskard.llm.client.bedrock import ClaudeBedrockClient
from giskard.llm.client.gemini import GeminiClient
from giskard.llm.client.groq_client import GroqClient
from giskard.llm.client.openai import OpenAIClient

PYDANTIC_V2 = pydantic.__version__.startswith("2.")

DEMO_OPENAI_RESPONSE = ChatCompletion(
    id="chatcmpl-abc123",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(content="This is a test!", role="assistant", function_call=None),
            logprobs=None,
        )
    ],
    created=1677858242,
    model="gpt-3.5-turbo-0613",
    object="chat.completion",
    usage=CompletionUsage(completion_tokens=7, prompt_tokens=13, total_tokens=20),
)


@pytest.mark.skipif(not PYDANTIC_V2, reason="LiteLLM raise an error with pydantic < 2")
def test_llm_complete_message():
    from giskard.llm.client import ChatMessage

    client = Mock()
    client.chat.completions.create.return_value = DEMO_OPENAI_RESPONSE
    res = OpenAIClient("gpt-35-turbo", client).complete(
        [ChatMessage(role="system", content="Hello")], temperature=0.11, max_tokens=1
    )

    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args[1]["messages"] == [{"role": "system", "content": "Hello"}]
    assert client.chat.completions.create.call_args[1]["temperature"] == 0.11
    assert client.chat.completions.create.call_args[1]["max_tokens"] == 1

    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"


@patch("litellm.completion")
@pytest.mark.skipif(not PYDANTIC_V2, reason="LiteLLM raise an error with pydantic < 2")
def test_litellm_client(completion):
    from giskard.llm.client import ChatMessage
    from giskard.llm.client.litellm import LiteLLMClient

    completion.return_value = DEMO_OPENAI_RESPONSE
    client = Mock()
    client.chat.completions.create.return_value = DEMO_OPENAI_RESPONSE
    llm_client = LiteLLMClient("gpt-4o", True, completion_params={"api_key": "api_key"})
    cfg = llm_client.get_config()
    assert cfg == {
        "client_type": "LiteLLMClient",
        "model": "gpt-4o",
        "disable_structured_output": True,
        "completion_params": {"api_key": "api_key"},
    }

    res = llm_client.complete([ChatMessage(role="system", content="Hello")], temperature=0.11, max_tokens=1)

    completion.assert_called_once()
    assert completion.call_args[1]["messages"] == [{"role": "system", "content": "Hello"}]
    assert completion.call_args[1]["temperature"] == 0.11
    assert completion.call_args[1]["max_tokens"] == 1
    assert completion.call_args[1]["api_key"] == "api_key"
    assert completion.call_args[1]["model"] == "gpt-4o"

    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"


API_KEY = "MOCK_API_KEY"


@pytest.mark.skipif(not PYDANTIC_V2, reason="LiteLLM raise an error with pydantic < 2")
def test_litellm_client_custom_model():
    import litellm

    from giskard.llm import get_default_client, set_llm_model
    from giskard.llm.client import ChatMessage

    class MockLLM(litellm.CustomLLM):
        def completion(self, model: str, messages: list, api_key: str, **kwargs) -> litellm.ModelResponse:
            assert api_key == API_KEY, "Completion params are not passed properly"

            return litellm.ModelResponse(
                choices=[
                    litellm.Choices(
                        model=model,
                        message=litellm.Message(
                            role="assistant", content=f"Mock response - {messages[-1].get('content')}"
                        ),
                    )
                ]
            )

    litellm.custom_provider_map = litellm.custom_provider_map + [{"provider": "mock", "custom_handler": MockLLM()}]

    set_llm_model("mock/faux-bot", api_key=API_KEY)

    llm_client = get_default_client()
    cfg = llm_client.get_config()
    assert cfg == {
        "client_type": "LiteLLMClient",
        "model": "mock/faux-bot",
        "disable_structured_output": False,
        "completion_params": {"api_key": API_KEY},
    }

    message = "Mock input"
    response = llm_client.complete([ChatMessage(role="user", content=message)])
    assert f"Mock response - {message}" == response.content


@pytest.mark.skipif(not PYDANTIC_V2, reason="LiteLLM raise an error with pydantic < 2")
def test_litellm_client_custom_model_with_embeddings():
    import litellm

    from giskard.llm import get_default_client, set_llm_model
    from giskard.llm.client import ChatMessage
    from giskard.llm.embeddings import get_default_embedding, set_embedding_model

    class MockLLMWithEmbeddings(litellm.CustomLLM):
        def completion(self, model: str, messages: list, api_key: str, **kwargs) -> litellm.ModelResponse:
            assert api_key == API_KEY, "Completion params are not passed properly"

            return litellm.ModelResponse(
                choices=[
                    litellm.Choices(
                        model=model,
                        message=litellm.Message(
                            role="assistant", content=f"Mock response - {messages[-1].get('content')}"
                        ),
                    )
                ]
            )

        def embedding(self, model: str, input: list, api_key: str, **kwargs) -> litellm.EmbeddingResponse:
            assert api_key == API_KEY, "Embedding params are not passed properly"

            embeddings_data = []
            for idx, _ in enumerate(input):
                embedding_vector = [float(idx + i) for i in range(10)]
                embeddings_data.append({"embedding": embedding_vector, "index": idx})

            return litellm.EmbeddingResponse(
                model=model,
                data=embeddings_data,
                object="list",
            )

    litellm.custom_provider_map = litellm.custom_provider_map + [
        {"provider": "mock-embed", "custom_handler": MockLLMWithEmbeddings()}
    ]

    set_llm_model("mock-embed/faux-bot", api_key=API_KEY)
    set_embedding_model("mock-embed/faux-bot", api_key=API_KEY)

    llm_client = get_default_client()
    cfg = llm_client.get_config()
    assert cfg == {
        "client_type": "LiteLLMClient",
        "model": "mock-embed/faux-bot",
        "disable_structured_output": False,
        "completion_params": {"api_key": API_KEY},
    }

    message = "Mock input"
    response = llm_client.complete([ChatMessage(role="user", content=message)])
    assert f"Mock response - {message}" == response.content

    embedding = get_default_embedding()
    assert embedding.model == "mock-embed/faux-bot"

    test_texts = ["Hello", "World"]
    embeddings = embedding.embed(test_texts)

    assert embeddings.shape == (2, 10)
    assert embeddings[0][0] == 0.0
    assert embeddings[1][0] == 1.0


@pytest.mark.skipif(not PYDANTIC_V2, reason="Mistral raise an error with pydantic < 2")
def test_mistral_client():
    from mistralai.models import ChatCompletionChoice, ChatCompletionResponse, UsageInfo

    demo_response = ChatCompletionResponse(
        id="2d62260a7a354e02922a4f6ad36930d3",
        object="chat.completion",
        created=1630000000,
        model="mistral-large",
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": "This is a test!"},
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=9, total_tokens=89, completion_tokens=80),
    )

    client = Mock()
    client.chat.complete.return_value = demo_response

    from giskard.llm.client.mistral import MistralClient

    llm_client = MistralClient(model="mistral-large", client=client)
    cfg = llm_client.get_config()
    assert cfg == {"client_type": "MistralClient", "model": "mistral-large"}

    res = llm_client.complete([ChatMessage(role="user", content="Hello")], temperature=0.11, max_tokens=12)

    client.chat.complete.assert_called_once()
    assert client.chat.complete.call_args[1]["messages"] == [{"role": "user", "content": "Hello"}]
    assert client.chat.complete.call_args[1]["temperature"] == 0.11
    assert client.chat.complete.call_args[1]["max_tokens"] == 12

    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"


def test_claude_bedrock_client():
    # Mock the bedrock_runtime_client
    bedrock_runtime_client = Mock()
    bedrock_runtime_client.invoke_model = MagicMock(
        return_value={
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps(
                        {
                            "id": "chatcmpl-abc123",
                            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "This is a test!"}],
                            "stop_reason": "end_turn",
                            "usage": {
                                "input_tokens": 9,
                                "output_tokens": 89,
                            },
                        }
                    )
                )
            )
        }
    )

    # Initialize the ClaudeBedrockClient with the mocked bedrock_runtime_client
    client = ClaudeBedrockClient(
        bedrock_runtime_client, model="anthropic.claude-3-sonnet-20240229-v1:0", anthropic_version="bedrock-2023-05-31"
    )
    cfg = client.get_config()
    assert cfg == {"client_type": "ClaudeBedrockClient", "model": "anthropic.claude-3-sonnet-20240229-v1:0"}

    # Call the complete method
    res = client.complete([ChatMessage(role="user", content="Hello")], temperature=0.11, max_tokens=12)

    # Assert that the invoke_model method was called with the correct arguments
    bedrock_runtime_client.invoke_model.assert_called_once()

    # Assert that the response is a ChatMessage and has the correct content
    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"


def test_gemini_client():
    # Mock the Gemini client
    gemini_api_client = Mock()
    gemini_api_client.generate_content = MagicMock(
        return_value=Mock(text="This is a test!", candidates=[Mock(content=Mock(role="assistant"))])
    )
    gemini_api_client.count_tokens = MagicMock(
        side_effect=lambda text: Mock(
            total_tokens=sum(len(t.split()) for t in text) if isinstance(text, list) else len(text.split())
        )
    )

    # Initialize the GeminiClient with the mocked gemini_api_client
    client = GeminiClient(model="gemini-pro", _client=gemini_api_client)
    cfg = client.get_config()
    assert cfg == {"client_type": "GeminiClient", "model": "gemini-pro"}

    # Call the complete method
    res = client.complete([ChatMessage(role="user", content="Hello")], temperature=0.11, max_tokens=12)
    print(res)

    # Assert that the generate_content method was called with the correct arguments
    gemini_api_client.generate_content.assert_called_once()
    assert gemini_api_client.generate_content.call_args[1]["contents"] == ([ContentDict(role="user", parts=["Hello"])])
    assert gemini_api_client.generate_content.call_args[1]["generation_config"].temperature == 0.11
    assert gemini_api_client.generate_content.call_args[1]["generation_config"].max_output_tokens == 12

    # Assert that the response is a ChatMessage and has the correct content
    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"


# Check if groq is installed
has_groq = find_spec("groq") is not None


@pytest.mark.skipif(not PYDANTIC_V2 or not has_groq, reason="Groq client test requires Pydantic v2 and groq package")
def test_groq_client():
    # Mock the Groq response
    demo_response = Mock()
    demo_response.usage = Mock(prompt_tokens=13, completion_tokens=7)
    demo_response.choices = [Mock(message=Mock(role="assistant", content="This is a test!"))]

    # Mock the Groq client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = demo_response

    client = GroqClient(model="llama-3.3-70b-versatile", client=mock_client)

    # Call the complete method
    res = client.complete([ChatMessage(role="user", content="Hello")], temperature=0.7, format="json", max_tokens=100)

    # Assert json_object format was passed
    assert mock_client.chat.completions.create.call_args[1]["response_format"] == {"type": "json_object"}

    # Assert that create was called with correct arguments
    mock_client.chat.completions.create.assert_called_once()
    assert mock_client.chat.completions.create.call_args[1]["messages"] == [{"role": "user", "content": "Hello"}]
    assert mock_client.chat.completions.create.call_args[1]["temperature"] == 0.7
    assert mock_client.chat.completions.create.call_args[1]["max_tokens"] == 100
    assert mock_client.chat.completions.create.call_args[1]["model"] == "llama-3.3-70b-versatile"

    # Assert the response is correct
    assert isinstance(res, ChatMessage)
    assert res.content == "This is a test!"
