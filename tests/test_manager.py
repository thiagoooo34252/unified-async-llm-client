from collections.abc import AsyncIterator, Sequence

import pytest
from pydantic import SecretStr

from clients.base import BaseLLMClient
from manager import AsyncLLMManager
from schemas import (
    ChatMessage,
    GenerationResult,
    LLMConfig,
    MessageRole,
    ModelResponse,
    Provider,
    StreamEvent,
)


class FakeClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.closed = False

    async def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            content=messages[-1].content,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(content=messages[-1].content)
        yield StreamEvent(done=True)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_manager_exposes_common_generate_and_stream_contract() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("test-key"),
    )
    fake = FakeClient(config)
    messages = [ChatMessage(role=MessageRole.USER, content="hello")]

    async with AsyncLLMManager(config, client=fake) as manager:
        response = await manager.generate(messages)
        events = [event async for event in manager.stream(messages)]

    assert isinstance(response, ModelResponse)
    assert response.content == "hello"
    assert [event.content for event in events] == ["hello", None]
    assert events[-1].done is True
    assert fake.closed is True
