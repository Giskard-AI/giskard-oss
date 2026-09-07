import re

# OpenAI/Azure reject schema names not matching ``^[a-zA-Z0-9_-]+$`` (e.g. the
# brackets parametrized generics like ``LLMGeneratorOutput[str]`` carry).
_INVALID_SCHEMA_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_schema_name(name: str) -> str:
    """Replace characters a provider would reject in a JSON-schema name."""
    return _INVALID_SCHEMA_NAME_CHARS.sub("_", name)
