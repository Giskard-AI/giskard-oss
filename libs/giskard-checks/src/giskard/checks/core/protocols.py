from collections.abc import AsyncGenerator
from typing import Protocol


class InteractionGenerator[YieldType, SendType](Protocol):
    def generate(self, trace: SendType) -> AsyncGenerator[YieldType, SendType]: ...
