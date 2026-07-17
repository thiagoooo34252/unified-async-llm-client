from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from clients.base import BaseLLMClient
from schemas import (
    ChatMessage,
    ErrorResponse,
    ErrorType,
    GenerationResult,
    LLMConfig,
    MessageRole,
    ModelResponse,
    StreamEvent,
)


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        config: LLMConfig,
        client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client or AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )

    async def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        system, api_messages = self._normalize_messages(messages)
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                system=system,
                messages=api_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except anthropic.RateLimitError:
            return self.error(
                ErrorType.RATE_LIMIT,
                "Anthropic rate limit reached after retries",
                retryable=True,
            )
        except anthropic.APITimeoutError:
            return self.error(
                ErrorType.TIMEOUT,
                "Anthropic request timed out after retries",
                retryable=True,
            )
        except anthropic.APIConnectionError:
            return self.error(
                ErrorType.CONNECTION,
                "Anthropic could not be reached after retries",
                retryable=True,
            )
        except anthropic.APIStatusError as exc:
            return self._status_error(exc)

        content = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not content:
            return self.error(
                ErrorType.EMPTY_RESPONSE,
                "Anthropic returned no text content",
                retryable=False,
            )
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[StreamEvent]:
        system, api_messages = self._normalize_messages(messages)
        emitted_content = False
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                system=system,
                messages=api_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        emitted_content = True
                        yield StreamEvent(content=text)
        except anthropic.RateLimitError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.RATE_LIMIT,
                    "Anthropic rate limit reached during streaming",
                    retryable=True,
                )
            )
            return
        except anthropic.APITimeoutError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.TIMEOUT,
                    "Anthropic streaming request timed out",
                    retryable=True,
                )
            )
            return
        except anthropic.APIConnectionError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.CONNECTION,
                    "Anthropic streaming connection failed",
                    retryable=True,
                )
            )
            return
        except anthropic.APIStatusError as exc:
            yield StreamEvent(error=self._status_error(exc))
            return
        if not emitted_content:
            yield StreamEvent(
                error=self.error(
                    ErrorType.EMPTY_RESPONSE,
                    "Anthropic stream returned no text content",
                    retryable=False,
                )
            )
            return
        yield StreamEvent(done=True)

    async def close(self) -> None:
        await self._client.close()

    def _normalize_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, list[MessageParam]]:
        validated = self.validate_messages(messages)
        system = "\n\n".join(
            message.content
            for message in validated
            if message.role is MessageRole.SYSTEM
        )
        api_messages = [
            MessageParam(
                role=message.role.value,
                content=message.content,
            )
            for message in validated
            if message.role is not MessageRole.SYSTEM
        ]
        return system, api_messages

    def _status_error(self, error: anthropic.APIStatusError) -> ErrorResponse:
        retryable = error.status_code >= 500
        return self.error(
            ErrorType.API,
            f"Anthropic returned HTTP {error.status_code}",
            retryable=retryable,
        )
