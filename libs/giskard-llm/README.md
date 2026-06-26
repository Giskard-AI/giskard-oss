# giskard-llm

Lightweight LLM routing over native provider SDKs. Routes `provider/model` strings to OpenAI, Google Gemini, Anthropic, Azure OpenAI, and Azure AI Foundry.

```bash
pip install giskard-llm[openai]      # OpenAI + Azure OpenAI + Azure AI Foundry
pip install giskard-llm[google]      # Google Gemini
pip install giskard-llm[anthropic]   # Anthropic
pip install giskard-llm[all]         # All providers
```

Requires Python 3.12+.

For architecture and design rationale, see [docs/design.md](docs/design.md). Per-provider behavior (role mapping, errors, kwargs) is documented in provider class docstrings under `src/giskard/llm/providers/`.

## Quick start

```python
from giskard.llm import acompletion

response = await acompletion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Provider reference

| Prefix | SDK | Auth env var | Completion | Embeddings |
|---|---|---|---|---|
| `openai/` (default) | `openai` | `OPENAI_API_KEY` | yes | yes |
| `google/` | `google-genai` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes | yes |
| `anthropic/` | `anthropic` | `ANTHROPIC_API_KEY` | yes | no |
| `azure/` | `openai` | `AZURE_API_KEY`, `AZURE_API_BASE` | yes | yes |
| `azure_ai/` | `openai` | `AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT` | yes | model-dependent |

## Development

From the repository root — see [CONTRIBUTING.md](../../CONTRIBUTING.md):

```bash
make test-unit PACKAGE=giskard-llm
```
