# 🤖 Setting up the LLM Client

This guide focuses primarily on configuring and using various LLM clients supported to run Giskard's LLM-assisted functionalities. We are using [LiteLLM](https://github.com/BerriAI/litellm) to handle the model calls, you can see the list of supported models in the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

## OpenAI Client Setup

More information on [OpenAI LiteLLM documentation](https://docs.litellm.ai/docs/providers/openai)

### Setup using .env variables

```python
import os
import giskard

os.environ["OPENAI_API_KEY"] = "" # "my-openai-api-key"

# Optional, setup a model (default LLM is gpt-4o, default embedding model is text-embedding-3-small)
giskard.llm.set_llm_model("gpt-4o")
giskard.llm.set_embedding_model("text-embedding-3-small")

# Optional Keys - OpenAI Organization, OpenAI API Base
os.environ["OPENAI_ORGANIZATION"] = "" # "my-openai-organization"
os.environ["OPENAI_API_BASE"] = "" # "https://api.openai.com"
```

### Setup using completion params

```python
import giskard

api_key = "" # "my-openai-api-key"

giskard.llm.set_llm_model("o1-preview", api_key=api_key)
giskard.llm.set_embedding_model("text-embedding-3-large", api_key=api_key)
```

## Azure OpenAI Client Setup

More information on [Azure LiteLLM documentation](https://docs.litellm.ai/docs/providers/azure)

### Setup using .env variables

```python
import os
import giskard

os.environ["AZURE_API_KEY"] = "" # "my-azure-api-key"
os.environ["AZURE_API_BASE"] = "" # "https://example-endpoint.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "" # "2023-05-15"

giskard.llm.set_llm_model("azure/<your_llm_name>")
giskard.llm.set_embedding_model("azure/<your_embed_model_name>")

# Optional Keys - Azure AD Token, Azure API Type
os.environ["AZURE_AD_TOKEN"] = ""
os.environ["AZURE_API_TYPE"] = ""
```

### Setup using completion params

```python
import giskard

api_base = "" # "https://example-endpoint.openai.azure.com"
api_version = "" # "2023-05-15"

# Using api_key
api_key = "" # "my-azure-api-key"
giskard.llm.set_llm_model("azure/<your_llm_name>", api_base=api_base, api_version=api_version, api_key=api_key)
giskard.llm.set_embedding_model("azure/<your_embed_model_name>", api_base=api_base, api_version=api_version, api_key=api_key)

# Using azure_ad_token
azure_ad_token = "" # "my-azure-ad-token"
giskard.llm.set_llm_model("azure/<your_llm_name>", api_base=api_base, api_version=api_version, azure_ad_token=azure_ad_token)
giskard.llm.set_embedding_model("azure/<your_embed_model_name>", api_base=api_base, api_version=api_version, azure_ad_token=azure_ad_token)
```

## Mistral Client Setup

More information on [Mistral LiteLLM documentation](https://docs.litellm.ai/docs/providers/mistral)

### Setup using .env variables

```python
import os
import giskard

os.environ["MISTRAL_API_KEY"] = "" # "my-mistral-api-key"

giskard.llm.set_llm_model("mistral/mistral-large-latest")
giskard.llm.set_embedding_model("mistral/mistral-embed")

```

## Ollama Client Setup

More information on [Ollama LiteLLM documentation](https://docs.litellm.ai/docs/providers/ollama)

### Setup using completion params

```python
import giskard

api_base = "http://localhost:11434" # default api_base for local Ollama

# See supported models here: https://docs.litellm.ai/docs/providers/ollama#ollama-models
giskard.llm.set_llm_model("ollama/qwen2.5", disable_structured_output=True, api_base=api_base)
giskard.llm.set_embedding_model("ollama/nomic-embed-text", api_base=api_base)
```

If you encounter errors with the embedding model in a Jupyter notebook, run this code:

```python
import nest_asyncio
nest_asyncio.apply()
```

## AWS Bedrock Client Setup

More information on [Bedrock LiteLLM documentation](https://docs.litellm.ai/docs/providers/bedrock)

### Setup using .env variables

```python
import os
import giskard

os.environ["AWS_ACCESS_KEY_ID"] = "" # "my-aws-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # "my-aws-secret-access-key"
os.environ["AWS_REGION_NAME"] = "" # "us-west-2"

giskard.llm.set_llm_model("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", disable_structured_output=True)
giskard.llm.set_embedding_model("bedrock/amazon.titan-embed-image-v1")
```

## Gemini Client Setup

More information on [Gemini LiteLLM documentation](https://docs.litellm.ai/docs/providers/gemini)

### Setup using .env variables

```python
import os
import giskard

os.environ["GEMINI_API_KEY"] = "" # "my-gemini-api-key"

giskard.llm.set_llm_model("gemini/gemini-1.5-pro")
giskard.llm.set_embedding_model("gemini/text-embedding-004")
```

## Groq Client Setup

More information on [Groq LiteLLM documentation](https://docs.litellm.ai/docs/providers/groq)

**Note: Groq does not currently support embedding models.**  
For a complete list of supported embedding providers, see: [LiteLLM Embedding Documentation](https://docs.litellm.ai/docs/embedding/supported_embedding)

### Setup using .env variables

```python
import os
import giskard

os.environ["GROQ_API_KEY"] = "" # "my-groq-api-key"

# Optional, setup a model (default LLM is llama-3.3-70b-versatile)
giskard.llm.set_llm_model("groq/llama-3.3-70b-versatile")
```

### Setup using completion params

```python
import giskard

api_key = "" # "my-groq-api-key"
giskard.llm.set_llm_model("groq/llama-3.3-70b-versatile", api_key=api_key)
```

## Custom Client Setup

More information on [Custom Format LiteLLM documentation](https://docs.litellm.ai/docs/providers/custom_llm_server)

```python
import os
import requests
from typing import Optional

import litellm
import giskard


class MyCustomLLM(litellm.CustomLLM):
    def completion(self, messages: str, api_key: Optional[str] = None, **kwargs) -> litellm.ModelResponse:
        api_key = api_key or os.environ.get("MY_SECRET_KEY")
        if api_key is None:
            raise litellm.AuthenticationError("`api_key` was not provided")

        response = requests.post(
            "https://www.my-custom-llm.ai/chat/completion",
            json={"messages": messages},
            headers={"Authorization": api_key},
        )

        return litellm.ModelResponse(**response.json())
    
    # Custom embeddings
    def embedding(
        self,
        model: str,
        input: list,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        api_key = api_key or os.environ.get("MY_SECRET_KEY")
        if api_key is None:
            raise litellm.AuthenticationError("`api_key` was not provided")

        response = requests.post(
            "https://www.my-custom-llm.ai/embeddings",
            json={"model": model, "input": input},
            headers={"Authorization": api_key},
        )

        return litellm.EmbeddingResponse(**response.json())

os.environ["MY_SECRET_KEY"] = "" # "my-secret-key"

my_custom_llm = MyCustomLLM()

litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
    {"provider": "my-custom-provider", "custom_handler": my_custom_llm}
]

api_key = os.environ["MY_SECRET_KEY"]

giskard.llm.set_llm_model("my-custom-provider/my-custom-model", api_key=api_key)
giskard.llm.set_embedding_model("my-custom-provider/my-custom-model", api_key=api_key)
```

If you run into any issues configuring the LLM client, don't hesitate to [ask us on Discord](https://discord.com/invite/ABvfpbu69R) or open a new issue on [our GitHub repo](https://github.com/Giskard-AI/giskard).
