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
client.configure("openai", api_key="sk-...")  # pragma: allowlist secret
client.configure(
    "azure-prod",
    provider="azure",
    api_key="os.environ/AZURE_PROD_KEY",  # pragma: allowlist secret
    base_url="os.environ/AZURE_PROD_ENDPOINT",
    api_version="2024-02-01",
)
client.configure(
    "anthropic-relaxed",
    provider="anthropic",
    api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    merge_system=True,
    schema_mutation="ignore",  # warn (default) | raise | ignore for structured-output schema changes
)

response = await client.acompletion("azure-prod/gpt-4o", messages)
response = await client.acompletion(
    "anthropic-relaxed/claude-3-5-haiku-latest", messages
)
```

## Provider reference

| Prefix | SDK | Auth env var | Completion | Embeddings | Notable kwargs |
|---|---|---|---|---|---|
| `google/` | `google-genai` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes | yes | `schema_mutation`, `http_client`, `default_headers`, `http_options` |
| `openai/` (default) | `openai` | `OPENAI_API_KEY` | yes | yes | `base_url`, `timeout`, `schema_mutation`, `http_client`, `default_headers` |
| `anthropic/` | `anthropic` | `ANTHROPIC_API_KEY` | yes | no | `merge_system`, `schema_mutation`, `timeout`, `http_client` (`httpx2`, see below), `default_headers` |
| `azure/` | `openai` | `AZURE_API_KEY`, `AZURE_API_BASE` | yes | yes | `api_version`, `base_url`, `schema_mutation`, `http_client`, `default_headers` |
| `azure_ai/` | `openai` | `AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT` | yes | model-dependent | `base_url`, `schema_mutation`, `http_client`, `default_headers` |

### Structured output (`response_format`)

Pass a Pydantic model as ``response_format`` on ``acompletion`` / ``aresponse`` (OpenAI-shaped API). Providers normalize JSON Schema before calling the SDK (nested ``$defs``, ``additionalProperties``, provider-specific keyword rules).

Configure ``schema_mutation`` on the provider (via ``LLMClient.configure``). It is **not** accepted on ``acompletion`` / ``aresponse`` kwargs (unknown params are stripped with a warning).

| Value | Behavior |
|---|---|
| ``warn`` (default) | Apply normalization; log **once per model and profile** when changes are *lossy* (see below). |
| ``raise`` | ``BadRequestError`` if lossy normalization would be required (message lists paths/keywords). |
| ``ignore`` | Normalize silently. |

**Lossy** means validation-relevant schema content changed: removed constraints (``pattern``, ``minimum``, …), stripped ``$ref`` siblings, dropped ``properties`` / ``$defs`` entries, etc. **Not lossy**: removing metadata-only keywords (``default``, ``title``, ``description``, … — Pydantic still parses and applies defaults) or adding ``additionalProperties: false`` alone.

Supported on ``openai``, ``anthropic``, ``google``, ``azure``, and ``azure_ai`` providers.


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
    api_key="os.environ/AZURE_OPENAI_API_KEY",  # pragma: allowlist secret
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
    api_key="os.environ/AZURE_AI_API_KEY",  # pragma: allowlist secret
    base_url="os.environ/AZURE_AI_ENDPOINT",
    http_client=http_client,
    default_headers={"x-ms-useragent": "giskard-llm"},
)
client.configure(
    "google-secure",
    provider="google",
    api_key="os.environ/GEMINI_API_KEY",  # pragma: allowlist secret
    http_client=http_client,
)

response = await client.acompletion("azure-secure/gpt-4.1-nano", messages)
await http_client.aclose()
```

The `anthropic` provider is the exception. Anthropic SDK v1 requires an
`httpx2.AsyncClient` and raises a `TypeError` for an `httpx.AsyncClient`.

```python
import httpx2

anthropic_http_client = httpx2.AsyncClient(verify="/path/to/ca.pem")
client.configure(
    "anthropic-secure",
    provider="anthropic",
    api_key="os.environ/ANTHROPIC_API_KEY",  # pragma: allowlist secret
    http_client=anthropic_http_client,
)
```

For detailed per-provider documentation (role mapping, message constraints, tool format, error mapping), see the provider class docstrings in `src/giskard/llm/providers/`.
