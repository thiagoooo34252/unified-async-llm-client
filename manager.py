from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import (
    ChatMessage,
    ChatRequest,
    GenerationResult,
    LLMConfig,
    Provider,
    StreamEvent,
)


class AsyncLLMManager:
    def __init__(
        self,
        config: LLMConfig,
        client: BaseLLMClient | None = None,
    ) -> None:
        self.config = config
        self._client = client or self._create_client(config)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        request = ChatRequest(messages=list(messages))
        return await self._client.generate(request.messages)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[StreamEvent]:
        request = ChatRequest(messages=list(messages))
        async for event in self._client.stream(request.messages):
            yield event

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _create_client(config: LLMConfig) -> BaseLLMClient:
        match config.provider:
            case Provider.OPENAI:
                return OpenAIClient(config)
            case Provider.ANTHROPIC:
                return AnthropicClient(config)
