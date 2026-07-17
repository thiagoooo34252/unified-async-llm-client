from collections.abc import AsyncIterator

import pytest

from clients.base import BaseLLMClient
from manager import AsyncLLMManager
from schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerationResult,
    Provider,
    StreamEvent,
    StreamEventType,
)


class FakeClient(BaseLLMClient):
    def __init__(self) -> None:
        self.closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResponse(
            provider=Provider.OPENAI,
            model="fake-model",
            text=request.prompt,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            provider=Provider.OPENAI,
            model="fake-model",
            type=StreamEventType.DELTA,
            delta=request.prompt,
        )

    async def close(self) -> None:
        self.closed = True


def test_manager_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="At least one"):
        AsyncLLMManager({})


@pytest.mark.asyncio
async def test_manager_routes_and_closes_clients() -> None:
    client = FakeClient()
    manager = AsyncLLMManager({Provider.OPENAI: client})
    request = GenerationRequest(prompt="Hello")

    response = await manager.generate(Provider.OPENAI, request)
    assert response.ok is True
    assert response.text == "Hello"

    events = [event async for event in manager.stream(Provider.OPENAI, request)]
    assert events[0].delta == "Hello"

    await manager.close()
    assert client.closed is True


def test_manager_rejects_unconfigured_provider() -> None:
    manager = AsyncLLMManager({Provider.OPENAI: FakeClient()})

    with pytest.raises(ValueError, match="not configured"):
        manager.stream(Provider.ANTHROPIC, GenerationRequest(prompt="Hello"))
