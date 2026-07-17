from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

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


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation backed by the asynchronous Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "gpt-5.6-luna",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._default_model = default_model
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        model = request.model or self._default_model
        try:
            response = await self._client.responses.create(
                model=model,
                input=request.prompt,
                instructions=request.system_prompt,
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            text = response.output_text.strip()
            if not text:
                return self._empty_response(model)

            return GenerationResponse(
                provider=Provider.OPENAI,
                model=response.model or model,
                text=text,
                input_tokens=response.usage.input_tokens if response.usage else None,
                output_tokens=response.usage.output_tokens if response.usage else None,
            )
        except openai.AuthenticationError:
            return self._error(
                ErrorCode.AUTHENTICATION,
                "OpenAI rejected the configured API key.",
                retryable=False,
            )
        except openai.RateLimitError:
            return self._error(
                ErrorCode.RATE_LIMIT,
                "OpenAI rate limit reached. Try again later.",
                retryable=True,
            )
        except openai.APITimeoutError:
            return self._error(
                ErrorCode.TIMEOUT,
                "OpenAI did not respond before the timeout.",
                retryable=True,
            )
        except openai.APIConnectionError:
            return self._error(
                ErrorCode.CONNECTION,
                "Could not connect to OpenAI.",
                retryable=True,
            )
        except openai.APIStatusError as exc:
            return self._error(
                ErrorCode.PROVIDER,
                f"OpenAI returned HTTP {exc.status_code}.",
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
            stream = await self._client.responses.create(
                model=model,
                input=request.prompt,
                instructions=request.system_prompt,
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            async for event in stream:
                if event.type == "response.output_text.delta":
                    emitted_text = emitted_text or bool(event.delta)
                    yield StreamEvent(
                        provider=Provider.OPENAI,
                        model=model,
                        type=StreamEventType.DELTA,
                        delta=event.delta,
                    )
                elif event.type == "response.completed":
                    completed_model = event.response.model or model
                    if not emitted_text:
                        yield StreamEvent(
                            provider=Provider.OPENAI,
                            model=completed_model,
                            type=StreamEventType.ERROR,
                            error=self._empty_response(completed_model),
                        )
                        return
                    yield StreamEvent(
                        provider=Provider.OPENAI,
                        model=completed_model,
                        type=StreamEventType.COMPLETED,
                    )
        except openai.AuthenticationError:
            yield self._stream_error(
                model,
                ErrorCode.AUTHENTICATION,
                "OpenAI rejected the configured API key.",
                retryable=False,
            )
        except openai.RateLimitError:
            yield self._stream_error(
                model,
                ErrorCode.RATE_LIMIT,
                "OpenAI rate limit reached. Try again later.",
                retryable=True,
            )
        except openai.APITimeoutError:
            yield self._stream_error(
                model,
                ErrorCode.TIMEOUT,
                "OpenAI did not respond before the timeout.",
                retryable=True,
            )
        except openai.APIConnectionError:
            yield self._stream_error(
                model,
                ErrorCode.CONNECTION,
                "Could not connect to OpenAI.",
                retryable=True,
            )
        except openai.APIStatusError as exc:
            yield self._stream_error(
                model,
                ErrorCode.PROVIDER,
                f"OpenAI returned HTTP {exc.status_code}.",
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
            provider=Provider.OPENAI,
            code=code,
            message=message,
            retryable=retryable,
        )

    @classmethod
    def _empty_response(cls, model: str) -> ErrorResponse:
        return cls._error(
            ErrorCode.EMPTY_RESPONSE,
            f"OpenAI returned an empty response for model {model}.",
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
            provider=Provider.OPENAI,
            model=model,
            type=StreamEventType.ERROR,
            error=cls._error(code, message, retryable=retryable),
        )
