from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

from schemas import GenerationRequest, GenerationResult, StreamEvent


class BaseLLMClient(ABC):
    """Contract shared by every asynchronous LLM provider client."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a complete response."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        """Yield text deltas and a final completion or error event."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
