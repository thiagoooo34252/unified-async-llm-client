import pytest
from pydantic import ValidationError

from schemas import (
    ErrorCode,
    ErrorResponse,
    GenerationRequest,
    Provider,
    StreamEvent,
    StreamEventType,
)


def test_generation_request_rejects_blank_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="   ")


def test_generation_request_rejects_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="Hello", temperature=1.1)


def test_delta_event_requires_text() -> None:
    with pytest.raises(ValidationError, match="require text"):
        StreamEvent(
            provider=Provider.OPENAI,
            model="gpt-test",
            type=StreamEventType.DELTA,
        )


def test_error_event_requires_error_payload() -> None:
    with pytest.raises(ValidationError, match="require an error"):
        StreamEvent(
            provider=Provider.ANTHROPIC,
            model="claude-test",
            type=StreamEventType.ERROR,
        )


def test_error_event_accepts_normalized_error() -> None:
    error = ErrorResponse(
        provider=Provider.ANTHROPIC,
        code=ErrorCode.TIMEOUT,
        message="Timed out.",
        retryable=True,
    )

    event = StreamEvent(
        provider=Provider.ANTHROPIC,
        model="claude-test",
        type=StreamEventType.ERROR,
        error=error,
    )

    assert event.error == error
