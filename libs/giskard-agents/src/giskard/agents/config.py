import os

from pydantic import BaseModel


class GiskardAgentsSettings(BaseModel):
    # Max items for single embedding API call (for OpenAI API this is 2048)
    embeddings_max_batch_size: int = int(
        os.environ.get("EMBEDDINGS_MAX_BATCH_SIZE", "1024")
    )
    # Max chars for single embedding API call (OpenAI limit is 8192 tokens, ~25k chars)
    embeddings_max_total_chars: int = int(
        os.environ.get("EMBEDDINGS_MAX_TOTAL_CHARS", "20000")
    )


GISKARD_AGENTS_SETTINGS = GiskardAgentsSettings()
