# giskard-llm

Lightweight LLM routing layer over native provider SDKs. Routes `provider/model` strings to the correct async SDK (OpenAI, Google Gemini, Anthropic, Azure OpenAI, Azure AI Foundry).

## Installation

```bash
pip install giskard-llm[openai]      # OpenAI + Azure OpenAI + Azure AI Foundry
pip install giskard-llm[google]      # Google Gemini
pip install giskard-llm[anthropic]   # Anthropic
pip install giskard-llm[all]         # All providers
```

> **Note:** Azure OpenAI (`azure/`) and Azure AI Foundry (`azure_ai/`) use the `openai` SDK.
> Installing `giskard-llm[openai]` (or `giskard-llm[azure]`) covers all three.

## Quick start

```python
from giskard.llm import acompletion, aembedding

# Module-level functions use env vars automatically
response = await acompletion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Bare model names default to OpenAI
response = await acompletion(model="gpt-4o", messages=[...])
```

## LLMClient (programmatic configuration)

```python
from giskard.llm import LLMClient

client = LLMClient()

# Configure with explicit values or env var references
client.configure("openai", api_key="sk-...") # pragma: allowlist secret
client.configure("azure-prod", provider="azure",
    api_key="os.environ/AZURE_PROD_KEY", # pragma: allowlist secret
    base_url="os.environ/AZURE_PROD_ENDPOINT",
    api_version="2024-02-01",
)
client.configure("anthropic-relaxed", provider="anthropic",
    api_key="os.environ/ANTHROPIC_API_KEY", # pragma: allowlist secret
    merge_system=True,
)

response = await client.acompletion("azure-prod/gpt-4o", messages)
response = await client.acompletion("anthropic-relaxed/claude-3-5-haiku-latest", messages)
```

## Provider reference

| Prefix | SDK | Auth env var | Completion | Embeddings | Notable kwargs |
|---|---|---|---|---|---|
| `openai/` (default) | `openai` | `OPENAI_API_KEY` | yes | yes | `base_url`, `timeout` |
| `google/` | `google-genai` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes | yes | — |
| `anthropic/` | `anthropic` | `ANTHROPIC_API_KEY` | yes | no | `merge_system`, `timeout` |
| `azure/` | `openai` | `AZURE_API_KEY`, `AZURE_API_BASE` | yes | yes | `api_version`, `base_url` |
| `azure_ai/` | `openai` | `AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT` | yes | model-dependent | `base_url` |

For detailed per-provider documentation (role mapping, message constraints, tool format, error mapping), see the provider class docstrings in `src/giskard/llm/providers/`.
