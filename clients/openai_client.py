from collections.abc import AsyncIterator, Sequence
from typing import cast

import openai
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam

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


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        config: LLMConfig,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client or AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )

    async def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        instructions, input_messages = self._normalize_messages(messages)
        try:
            response = await self._client.responses.create(
                model=self.config.model,
                instructions=instructions or None,
                input=input_messages,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        except openai.RateLimitError:
            return self.error(
                ErrorType.RATE_LIMIT,
                "OpenAI rate limit reached after retries",
                retryable=True,
            )
        except openai.APITimeoutError:
            return self.error(
                ErrorType.TIMEOUT,
                "OpenAI request timed out after retries",
                retryable=True,
            )
        except openai.APIConnectionError:
            return self.error(
                ErrorType.CONNECTION,
                "OpenAI could not be reached after retries",
                retryable=True,
            )
        except openai.APIStatusError as exc:
            return self._status_error(exc)

        content = response.output_text.strip()
        if not content:
            return self.error(
                ErrorType.EMPTY_RESPONSE,
                "OpenAI returned no text content",
                retryable=False,
            )
        usage = response.usage
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            content=content,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[StreamEvent]:
        instructions, input_messages = self._normalize_messages(messages)
        emitted_content = False
        try:
            async with await self._client.responses.create(
                model=self.config.model,
                instructions=instructions or None,
                input=input_messages,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                stream=True,
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta" and event.delta:
                        emitted_content = True
                        yield StreamEvent(content=event.delta)
        except openai.RateLimitError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.RATE_LIMIT,
                    "OpenAI rate limit reached during streaming",
                    retryable=True,
                )
            )
            return
        except openai.APITimeoutError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.TIMEOUT,
                    "OpenAI streaming request timed out",
                    retryable=True,
                )
            )
            return
        except openai.APIConnectionError:
            yield StreamEvent(
                error=self.error(
                    ErrorType.CONNECTION,
                    "OpenAI streaming connection failed",
                    retryable=True,
                )
            )
            return
        except openai.APIStatusError as exc:
            yield StreamEvent(error=self._status_error(exc))
            return
        if not emitted_content:
            yield StreamEvent(
                error=self.error(
                    ErrorType.EMPTY_RESPONSE,
                    "OpenAI stream returned no text content",
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
    ) -> tuple[str, ResponseInputParam]:
        validated = self.validate_messages(messages)
        instructions = "\n\n".join(
            message.content
            for message in validated
            if message.role is MessageRole.SYSTEM
        )
        input_messages = cast(
            ResponseInputParam,
            [
                {"role": message.role.value, "content": message.content}
                for message in validated
                if message.role is not MessageRole.SYSTEM
            ],
        )
        return instructions, input_messages

    def _status_error(self, error: openai.APIStatusError) -> ErrorResponse:
        retryable = error.status_code >= 500
        return self.error(
            ErrorType.API,
            f"OpenAI returned HTTP {error.status_code}",
            retryable=retryable,
        )
