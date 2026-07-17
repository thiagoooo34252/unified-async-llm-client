from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import openai
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr

from clients import AnthropicClient, OpenAIClient
from schemas import (
    ChatMessage,
    ErrorResponse,
    ErrorType,
    LLMConfig,
    MessageRole,
    ModelResponse,
    Provider,
)


class FakeOpenAIStream:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def __aenter__(self) -> "FakeOpenAIStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self.events:
            yield event


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stream_events: list[Any] = [
            SimpleNamespace(
                type="response.output_text.delta",
                delta="streamed",
            )
        ]

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            return FakeOpenAIStream(self.stream_events)
        return SimpleNamespace(
            output_text="normal response",
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeAnthropicStream:
    def __init__(
        self,
        chunks: list[str],
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error

    async def __aenter__(self) -> "FakeAnthropicStream":
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def iterate() -> AsyncIterator[str]:
            for chunk in self.chunks:
                yield chunk

        return iterate()


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stream_error: Exception | None = None
        self.stream_chunks = ["streamed"]

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="normal response")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )

    def stream(self, **kwargs: Any) -> FakeAnthropicStream:
        self.calls.append(kwargs)
        return FakeAnthropicStream(self.stream_chunks, self.stream_error)


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="Be concise"),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]


@pytest.mark.asyncio
async def test_openai_normalizes_generate_and_stream_responses() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("test-key"),
    )
    fake = FakeOpenAI()
    client = OpenAIClient(config, client=cast(AsyncOpenAI, fake))

    response = await client.generate(messages())
    events = [event async for event in client.stream(messages())]
    await client.close()

    assert isinstance(response, ModelResponse)
    assert response.content == "normal response"
    assert response.input_tokens == 4
    assert [event.content for event in events] == ["streamed", None]
    assert fake.responses.calls[0]["instructions"] == "Be concise"
    assert fake.responses.calls[0]["input"] == [{"role": "user", "content": "Hello"}]
    assert fake.closed is True


@pytest.mark.asyncio
async def test_openai_returns_controlled_rate_limit_error() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("test-key"),
    )
    fake = FakeOpenAI()
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)
    fake.responses.error = openai.RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )
    client = OpenAIClient(config, client=cast(AsyncOpenAI, fake))

    result = await client.generate(messages())

    assert isinstance(result, ErrorResponse)
    assert result.error_type is ErrorType.RATE_LIMIT
    assert result.retryable is True
    assert "test-key" not in result.message


@pytest.mark.asyncio
async def test_openai_returns_controlled_authentication_error() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("test-key"),
    )
    fake = FakeOpenAI()
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(401, request=request)
    fake.responses.error = openai.AuthenticationError(
        "invalid key",
        response=response,
        body=None,
    )
    client = OpenAIClient(config, client=cast(AsyncOpenAI, fake))

    result = await client.generate(messages())

    assert isinstance(result, ErrorResponse)
    assert result.error_type is ErrorType.API
    assert result.retryable is False
    assert result.message == "OpenAI returned HTTP 401"
    assert "test-key" not in result.message


@pytest.mark.asyncio
async def test_anthropic_normalizes_generate_and_stream_responses() -> None:
    config = LLMConfig(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("test-key"),
    )
    fake = FakeAnthropic()
    client = AnthropicClient(config, client=cast(AsyncAnthropic, fake))

    response = await client.generate(messages())
    events = [event async for event in client.stream(messages())]
    await client.close()

    assert isinstance(response, ModelResponse)
    assert response.content == "normal response"
    assert response.output_tokens == 3
    assert [event.content for event in events] == ["streamed", None]
    assert fake.messages.calls[0]["system"] == "Be concise"
    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert fake.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC])
async def test_empty_stream_returns_controlled_error(provider: Provider) -> None:
    if provider is Provider.OPENAI:
        config = LLMConfig(
            provider=provider,
            openai_api_key=SecretStr("test-key"),
        )
        fake_openai = FakeOpenAI()
        fake_openai.responses.stream_events = []
        client = OpenAIClient(config, client=cast(AsyncOpenAI, fake_openai))
    else:
        config = LLMConfig(
            provider=provider,
            anthropic_api_key=SecretStr("test-key"),
        )
        fake_anthropic = FakeAnthropic()
        fake_anthropic.messages.stream_chunks = []
        client = AnthropicClient(
            config,
            client=cast(AsyncAnthropic, fake_anthropic),
        )

    events = [event async for event in client.stream(messages())]

    assert len(events) == 1
    assert events[0].error is not None
    assert events[0].error.error_type is ErrorType.EMPTY_RESPONSE
    assert events[0].done is False


@pytest.mark.asyncio
async def test_openai_stream_returns_controlled_timeout_error() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("test-key"),
    )
    fake = FakeOpenAI()
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake.responses.error = openai.APITimeoutError(request=request)
    client = OpenAIClient(config, client=cast(AsyncOpenAI, fake))

    events = [event async for event in client.stream(messages())]

    assert events[0].error is not None
    assert events[0].error.error_type is ErrorType.TIMEOUT
    assert events[0].error.retryable is True


@pytest.mark.asyncio
async def test_anthropic_stream_returns_controlled_connection_error() -> None:
    config = LLMConfig(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("test-key"),
    )
    fake = FakeAnthropic()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake.messages.stream_error = anthropic.APIConnectionError(request=request)
    client = AnthropicClient(config, client=cast(AsyncAnthropic, fake))

    events = [event async for event in client.stream(messages())]

    assert events[0].error is not None
    assert events[0].error.error_type is ErrorType.CONNECTION
    assert events[0].error.retryable is True


@pytest.mark.asyncio
async def test_anthropic_generate_returns_controlled_rate_limit_error() -> None:
    config = LLMConfig(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("test-key"),
    )
    fake = FakeAnthropic()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    fake.messages.error = anthropic.RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )
    client = AnthropicClient(config, client=cast(AsyncAnthropic, fake))

    result = await client.generate(messages())

    assert isinstance(result, ErrorResponse)
    assert result.error_type is ErrorType.RATE_LIMIT
    assert result.retryable is True
    assert "test-key" not in result.message


@pytest.mark.asyncio
async def test_anthropic_returns_controlled_authentication_error() -> None:
    config = LLMConfig(
        provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("test-key"),
    )
    fake = FakeAnthropic()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    fake.messages.error = anthropic.AuthenticationError(
        "invalid key",
        response=response,
        body=None,
    )
    client = AnthropicClient(config, client=cast(AsyncAnthropic, fake))

    result = await client.generate(messages())

    assert isinstance(result, ErrorResponse)
    assert result.error_type is ErrorType.API
    assert result.retryable is False
    assert result.message == "Anthropic returned HTTP 401"
    assert "test-key" not in result.message
