from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

from schemas import (
    ChatMessage,
    ChatRequest,
    ErrorResponse,
    ErrorType,
    GenerationResult,
    LLMConfig,
    StreamEvent,
)


class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @staticmethod
    def validate_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        return ChatRequest(messages=list(messages)).messages

    def error(
        self,
        error_type: ErrorType,
        message: str,
        *,
        retryable: bool,
    ) -> ErrorResponse:
        return ErrorResponse(
            provider=self.config.provider,
            model=self.config.model,
            error_type=error_type,
            message=message,
            retryable=retryable,
        )
