from counterpoint import Message

from giskard_checks.interactions.structured import StructuredInteraction

"""Chat interaction specialization.

`ChatInteraction` narrows the generic `Interaction` to scenarios where the input and output are messages.
"""


class ChatInteraction(StructuredInteraction[list[Message], list[Message]]):
    """Chat interaction."""
