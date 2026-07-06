# giskard-llm

Lightweight LLM routing layer over native provider SDKs. Routes `provider/model` strings to the correct async SDK (OpenAI, Google Gemini, Anthropic, Azure OpenAI, Azure AI Foundry, Eden AI).

## Installation

```bash
pip install giskard-llm[openai]      # OpenAI + Azure OpenAI + Azure AI Foundry
pip install giskard-llm[google]      # Google Gemini
pip install giskard-llm[anthropic]   # Anthropic
pip install giskard-llm[all]         # All providers
```

> **Note:** Azure OpenAI (`azure/`), Azure AI Foundry (`azure_ai/`) and Eden AI (`edenai/`) use the `openai` SDK.
> Installing `giskard-llm[openai]` covers all of them.

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
| `openai/` (default) | `openai` | `OPENAI_API_KEY` | yes | yes | `base_url`, `timeout`, `http_client`, `default_headers` |
| `google/` | `google-genai` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes | yes | `http_client`, `default_headers`, `http_options` |
| `anthropic/` | `anthropic` | `ANTHROPIC_API_KEY` | yes | no | `merge_system`, `timeout`, `http_client`, `default_headers` |
| `azure/` | `openai` | `AZURE_API_KEY`, `AZURE_API_BASE` | yes | yes | `api_version`, `base_url`, `http_client`, `default_headers` |
| `azure_ai/` | `openai` | `AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT` | yes | model-dependent | `base_url`, `http_client`, `default_headers` |
| `edenai/` | `openai` | `EDENAI_API_KEY` | yes | yes | `base_url` (default `https://api.edenai.run/v3`), `timeout`, `http_client`, `default_headers` |


## Azure Foundry OpenAI v1

Azure Foundry OpenAI v1 endpoints are OpenAI-compatible. Configure them with
the `openai` provider and the Azure `/openai/v1/` base URL. Use Azure
deployment names as the chat, response, and embedding model names.

```python
from giskard.llm import LLMClient

client = LLMClient()
client.configure(
    "foundry-v1",
    provider="openai",
    api_key="os.environ/AZURE_OPENAI_API_KEY", # pragma: allowlist secret
    base_url="https://example.openai.azure.com/openai/v1/",
)

chat = await client.acompletion(
    "foundry-v1/gpt-4.1-mini",
    [{"role": "user", "content": "Write one sentence."}],
)
response = await client.aresponse("foundry-v1/gpt-4.1-mini", "Write one sentence.")
embedding = await client.aembedding(
    "foundry-v1/text-embedding-3-small",
    ["Text to embed."],
)
```

Use `azure/` for classic Azure OpenAI deployments that require `api_version`.
Use `azure_ai/` for the existing Azure AI Foundry compatibility path. Do not
use `azure_ai/` for new OpenAI v1 endpoints unless you intentionally need that
legacy endpoint behavior.

## Eden AI (500+ models, one key, EU/GDPR)

[Eden AI](https://www.edenai.co/) is a unified, EU-based gateway exposing 500+ models
from every major provider (OpenAI, Anthropic, Google, Mistral, Amazon, xAI, DeepSeek,
Groq, …) behind a single OpenAI-compatible endpoint. The `edenai/` provider is a thin
subclass of the `openai/` provider with the base URL and `EDENAI_API_KEY` preset, so
completions, embeddings, and the Responses API all work.

Eden AI model ids use a `provider/model` form, so full giskard model strings keep both
segments (the router splits on the first `/` only):

```python
from giskard.llm import acompletion, aembedding

# EDENAI_API_KEY is read from the environment
chat = await acompletion(
    "edenai/openai/gpt-4o",
    [{"role": "user", "content": "Hello!"}],
)
chat = await acompletion(
    "edenai/anthropic/claude-3-5-sonnet-latest",
    [{"role": "user", "content": "Hello!"}],
)
emb = await aembedding("edenai/openai/text-embedding-3-small", ["Text to embed."])
```

### EU data residency (GDPR)

For strict data residency, use Eden AI's **dedicated EU endpoint**
(`https://api.eu.edenai.run/v3`) via `eu=True`. It serves only EU-hosted models
(~270) and **rejects any non-EU model with HTTP 451**, guaranteeing inference
stays within the European Union.

```python
from giskard.llm import LLMClient

client = LLMClient()
client.configure("eden-eu", provider="edenai", eu=True, api_key="os.environ/EDENAI_API_KEY")  # pragma: allowlist secret

# EU-hosted model → OK; a non-EU model would raise a 451 error.
chat = await client.acompletion("eden-eu/amazon/amazon.nova-lite-v1:0", messages)
```

Alternatively, keep the default endpoint and select EU-hosted model ids
yourself: `GET https://api.edenai.run/v3/models` returns a `regions` list per
model, and some providers publish explicit `@eu` variants (e.g.
`edenai/amazon/amazon.nova-2-lite-v1:0@eu`).

## Custom transport and headers

Use `http_client` to provide a caller-owned async HTTP client, for example
when your environment requires a custom CA bundle. giskard-llm passes this
client through to provider SDKs and does not close it.

```python
import httpx
from giskard.llm import LLMClient

http_client = httpx.AsyncClient(verify="/path/to/ca.pem")

client = LLMClient()
client.configure(
    "azure-secure",
    provider="azure_ai",
    api_key="os.environ/AZURE_AI_API_KEY", # pragma: allowlist secret
    base_url="os.environ/AZURE_AI_ENDPOINT",
    http_client=http_client,
    default_headers={"x-ms-useragent": "giskard-llm"},
)
client.configure(
    "google-secure",
    provider="google",
    api_key="os.environ/GEMINI_API_KEY", # pragma: allowlist secret
    http_client=http_client,
)

response = await client.acompletion("azure-secure/gpt-4.1-nano", messages)
await http_client.aclose()
```

For detailed per-provider documentation (role mapping, message constraints, tool format, error mapping), see the provider class docstrings in `src/giskard/llm/providers/`.
