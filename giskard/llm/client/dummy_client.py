from .base import LLMClient, ChatMessage
from typing import Sequence, Optional

class DummyLLMClient(LLMClient):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        # Simple echo implementation
        last_message = messages[-1] if messages else ChatMessage(role="user", content="Hello")
        return ChatMessage(role="assistant", content=f"Echo: {last_message.content}")

    def get_config(self) -> dict:
        return {"name": "DummyLLMClient", "version": "1.0"}

if __name__ == "__main__":
    client = DummyLLMClient()
    messages = [ChatMessage(role="user", content="Hi there!")]
    response = client.complete(messages)
    print("Response:", response)
    print("Config:", client.get_config())
