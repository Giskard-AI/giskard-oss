# OSS-58: Local embedding provider for SemanticSimilarity

## Goal

Add a first-class SentenceTransformers embedding provider so `SemanticSimilarity` can run with local embeddings and without an OpenAI API key.

## Plan

1. Add a lazy-loading `SentenceTransformerEmbedding` implementation in `giskard-agents`.
2. Expose it from `giskard.agents.embeddings` and from `giskard.checks.utils.embeddings` for the issue's documented import path.
3. Add optional `local-embeddings` dependencies without changing the core install.
4. Document `set_default_embedding_model()` with local and custom provider examples.
5. Add unit tests that mock `sentence_transformers` so CI does not download models.

## Verification

- Run focused embedding provider tests.
- Run semantic similarity tests if the environment setup is available.

## Results

- Added `SentenceTransformerEmbedding` in `giskard-agents` with lazy optional dependency import and async-safe local encoding.
- Re-exported the provider from `giskard.checks.utils.embeddings` via the `giskard.agents` package root to respect package-boundary checks.
- Added `local-embeddings` optional extras for root, checks, and agents packages.
- Kept local embeddings out of the broad `all`/`full` install path because the current `giskard-scan` dependency set requires `huggingface-hub>=1`, while SentenceTransformers 3.x resolves through Transformers packages requiring `huggingface-hub<1`.
- Verified with focused semantic similarity tests and affected package unit test targets.
