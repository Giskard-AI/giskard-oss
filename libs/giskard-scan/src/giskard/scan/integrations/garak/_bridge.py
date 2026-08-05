from garak.attempt import Conversation


def _conv_uuid(conversation: Conversation) -> str | None:
    for turn in conversation.turns:
        notes = turn.content.notes
        if notes and notes.get("uuid"):
            return notes["uuid"]
    return None
