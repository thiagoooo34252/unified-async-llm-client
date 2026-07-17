from collections.abc import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from clients.base import BaseLLMClient
from schemas import (
    ErrorCode,
    ErrorResponse,
    GenerationRequest,
    GenerationResponse,
    GenerationResult,
    Provider,
    StreamEvent,
    StreamEventType,
)


class AnthropicClient(BaseLLMClient):
    """Anthropic implementation backed by the asynchronous Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-5",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._default_model = default_model
        self._client = client or AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        model = request.model or self._default_model
        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.prompt}],
                system=request.system_prompt or anthropic.NOT_GIVEN,
                temperature=request.temperature,
            )
            text = "".join(
                block.text for block in message.content if block.type == "text"
            ).strip()
            if not text:
                return self._empty_response(model)

            return GenerationResponse(
                provider=Provider.ANTHROPIC,
                model=message.model or model,
                text=text,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )
        except anthropic.AuthenticationError:
            return self._error(
                ErrorCode.AUTHENTICATION,
                "Anthropic rejected the configured API key.",
                retryable=False,
            )
        except anthropic.RateLimitError:
            return self._error(
                ErrorCode.RATE_LIMIT,
                "Anthropic rate limit reached. Try again later.",
                retryable=True,
            )
        except anthropic.APITimeoutError:
            return self._error(
                ErrorCode.TIMEOUT,
                "Anthropic did not respond before the timeout.",
                retryable=True,
            )
        except anthropic.APIConnectionError:
            return self._error(
                ErrorCode.CONNECTION,
                "Could not connect to Anthropic.",
                retryable=True,
            )
        except anthropic.APIStatusError as exc:
            return self._error(
                ErrorCode.PROVIDER,
                f"Anthropic returned HTTP {exc.status_code}.",
                retryable=exc.status_code >= 500,
            )
        except ValueError as exc:
            return self._error(
                ErrorCode.CONFIGURATION,
                str(exc),
                retryable=False,
            )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        model = request.model or self._default_model
        emitted_text = False
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.prompt}],
                system=request.system_prompt or anthropic.NOT_GIVEN,
                temperature=request.temperature,
            ) as stream:
                async for text in stream.text_stream:
                    emitted_text = emitted_text or bool(text)
                    yield StreamEvent(
                        provider=Provider.ANTHROPIC,
                        model=model,
                        type=StreamEventType.DELTA,
                        delta=text,
                    )
                final_message = await stream.get_final_message()
                completed_model = final_message.model or model
                if not emitted_text:
                    yield StreamEvent(
                        provider=Provider.ANTHROPIC,
                        model=completed_model,
                        type=StreamEventType.ERROR,
                        error=self._empty_response(completed_model),
                    )
                    return
                yield StreamEvent(
                    provider=Provider.ANTHROPIC,
                    model=completed_model,
                    type=StreamEventType.COMPLETED,
                )
        except anthropic.AuthenticationError:
            yield self._stream_error(
                model,
                ErrorCode.AUTHENTICATION,
                "Anthropic rejected the configured API key.",
                retryable=False,
            )
        except anthropic.RateLimitError:
            yield self._stream_error(
                model,
                ErrorCode.RATE_LIMIT,
                "Anthropic rate limit reached. Try again later.",
                retryable=True,
            )
        except anthropic.APITimeoutError:
            yield self._stream_error(
                model,
                ErrorCode.TIMEOUT,
                "Anthropic did not respond before the timeout.",
                retryable=True,
            )
        except anthropic.APIConnectionError:
            yield self._stream_error(
                model,
                ErrorCode.CONNECTION,
                "Could not connect to Anthropic.",
                retryable=True,
            )
        except anthropic.APIStatusError as exc:
            yield self._stream_error(
                model,
                ErrorCode.PROVIDER,
                f"Anthropic returned HTTP {exc.status_code}.",
                retryable=exc.status_code >= 500,
            )
        except ValueError as exc:
            yield self._stream_error(
                model,
                ErrorCode.CONFIGURATION,
                str(exc),
                retryable=False,
            )

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _error(
        code: ErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> ErrorResponse:
        return ErrorResponse(
            provider=Provider.ANTHROPIC,
            code=code,
            message=message,
            retryable=retryable,
        )

    @classmethod
    def _empty_response(cls, model: str) -> ErrorResponse:
        return cls._error(
            ErrorCode.EMPTY_RESPONSE,
            f"Anthropic returned an empty response for model {model}.",
            retryable=True,
        )

    @classmethod
    def _stream_error(
        cls,
        model: str,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> StreamEvent:
        return StreamEvent(
            provider=Provider.ANTHROPIC,
            model=model,
            type=StreamEventType.ERROR,
            error=cls._error(code, message, retryable=retryable),
        )
