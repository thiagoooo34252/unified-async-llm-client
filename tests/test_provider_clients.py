from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, AuthenticationError, RateLimitError
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import ValidationError

from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from schemas import ErrorCode, GenerationRequest, StreamEventType


@pytest.mark.asyncio
async def test_openai_generate_returns_validated_response() -> None:
    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(
        return_value=SimpleNamespace(
            output_text="  Entropy measures uncertainty.  ",
            model="gpt-test",
            usage=SimpleNamespace(input_tokens=5, output_tokens=4),
        )
    )
    client = OpenAIClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="What is entropy?"))

    assert result.ok is True
    assert result.text == "Entropy measures uncertainty."
    assert result.total_tokens == 9


@pytest.mark.asyncio
async def test_openai_generate_normalizes_rate_limit() -> None:
    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(
        side_effect=RateLimitError(
            "rate limited",
            response=SimpleNamespace(
                request=SimpleNamespace(),
                status_code=429,
                headers={},
            ),
            body=None,
        )
    )
    client = OpenAIClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="Hello"))

    assert result.ok is False
    assert result.code is ErrorCode.RATE_LIMIT
    assert result.retryable is True


@pytest.mark.asyncio
async def test_openai_generate_normalizes_authentication_error() -> None:
    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(
        side_effect=AuthenticationError(
            "invalid key",
            response=SimpleNamespace(
                request=SimpleNamespace(),
                status_code=401,
                headers={},
            ),
            body=None,
        )
    )
    client = OpenAIClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="Hello"))

    assert result.ok is False
    assert result.code is ErrorCode.AUTHENTICATION
    assert result.retryable is False


@pytest.mark.asyncio
async def test_openai_generate_rejects_empty_response() -> None:
    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(
        return_value=SimpleNamespace(output_text=" ", model="gpt-test", usage=None)
    )
    client = OpenAIClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="Hello"))

    assert result.ok is False
    assert result.code is ErrorCode.EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_openai_stream_yields_delta_and_completion() -> None:
    async def stream_events():
        yield ResponseTextDeltaEvent(
            content_index=0,
            delta="Entropy",
            item_id="item-1",
            logprobs=[],
            output_index=0,
            sequence_number=1,
            type="response.output_text.delta",
        )
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(model="gpt-test"),
        )

    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(return_value=stream_events())
    client = OpenAIClient(client=sdk_client)

    events = [
        event
        async for event in client.stream(GenerationRequest(prompt="Define entropy"))
    ]

    assert [event.type for event in events] == [
        StreamEventType.DELTA,
        StreamEventType.COMPLETED,
    ]
    assert events[0].delta == "Entropy"


@pytest.mark.asyncio
async def test_openai_stream_rejects_empty_response() -> None:
    async def stream_events():
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(model="gpt-test"),
        )

    sdk_client = MagicMock(spec=AsyncOpenAI)
    sdk_client.responses.create = AsyncMock(return_value=stream_events())
    client = OpenAIClient(client=sdk_client)

    events = [
        event async for event in client.stream(GenerationRequest(prompt="Hello"))
    ]

    assert len(events) == 1
    assert events[0].type is StreamEventType.ERROR
    assert events[0].error is not None
    assert events[0].error.code is ErrorCode.EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_anthropic_generate_returns_validated_response() -> None:
    sdk_client = MagicMock(spec=AsyncAnthropic)
    sdk_client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Entropy is disorder.")],
            model="claude-test",
            usage=SimpleNamespace(input_tokens=6, output_tokens=5),
        )
    )
    client = AnthropicClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="What is entropy?"))

    assert result.ok is True
    assert result.text == "Entropy is disorder."
    assert result.total_tokens == 11


@pytest.mark.asyncio
async def test_anthropic_generate_rejects_empty_response() -> None:
    sdk_client = MagicMock(spec=AsyncAnthropic)
    sdk_client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="internal")],
            model="claude-test",
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        )
    )
    client = AnthropicClient(client=sdk_client)

    result = await client.generate(GenerationRequest(prompt="Hello"))

    assert result.ok is False
    assert result.code is ErrorCode.EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_anthropic_stream_yields_delta_and_completion() -> None:
    class FakeStreamContext:
        async def __aenter__(self):
            async def text_stream():
                yield "Entropy"

            return SimpleNamespace(
                text_stream=text_stream(),
                get_final_message=AsyncMock(
                    return_value=SimpleNamespace(model="claude-test")
                ),
            )

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    sdk_client = MagicMock(spec=AsyncAnthropic)
    sdk_client.messages.stream.return_value = FakeStreamContext()
    client = AnthropicClient(client=sdk_client)

    events = [
        event
        async for event in client.stream(GenerationRequest(prompt="Define entropy"))
    ]

    assert [event.type for event in events] == [
        StreamEventType.DELTA,
        StreamEventType.COMPLETED,
    ]
    assert events[0].delta == "Entropy"


@pytest.mark.asyncio
async def test_anthropic_stream_rejects_empty_response() -> None:
    class FakeStreamContext:
        async def __aenter__(self):
            async def text_stream():
                if False:
                    yield ""

            return SimpleNamespace(
                text_stream=text_stream(),
                get_final_message=AsyncMock(
                    return_value=SimpleNamespace(model="claude-test")
                ),
            )

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    sdk_client = MagicMock(spec=AsyncAnthropic)
    sdk_client.messages.stream.return_value = FakeStreamContext()
    client = AnthropicClient(client=sdk_client)

    events = [
        event async for event in client.stream(GenerationRequest(prompt="Hello"))
    ]

    assert len(events) == 1
    assert events[0].type is StreamEventType.ERROR
    assert events[0].error is not None
    assert events[0].error.code is ErrorCode.EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_clients_close_underlying_sdk() -> None:
    openai_sdk = MagicMock(spec=AsyncOpenAI)
    openai_sdk.close = AsyncMock()
    anthropic_sdk = MagicMock(spec=AsyncAnthropic)
    anthropic_sdk.close = AsyncMock()

    openai_client = OpenAIClient(client=openai_sdk)
    anthropic_client = AnthropicClient(client=anthropic_sdk)

    await openai_client.close()
    await anthropic_client.close()

    openai_sdk.close.assert_awaited_once()
    anthropic_sdk.close.assert_awaited_once()


def test_injected_clients_preserve_sdk_protocol() -> None:
    openai_sdk = cast(AsyncOpenAI, MagicMock(spec=AsyncOpenAI))
    anthropic_sdk = cast(AsyncAnthropic, MagicMock(spec=AsyncAnthropic))

    assert OpenAIClient(client=openai_sdk)
    assert AnthropicClient(client=anthropic_sdk)
