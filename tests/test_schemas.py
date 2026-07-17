import pytest
from pydantic import SecretStr, ValidationError

from schemas import (
    ChatMessage,
    ChatRequest,
    LLMConfig,
    MessageRole,
    Provider,
    StreamEvent,
)


def test_config_requires_selected_provider_key() -> None:
    with pytest.raises(ValidationError, match="API key is not configured"):
        LLMConfig(provider=Provider.OPENAI)


def test_config_redacts_secrets() -> None:
    config = LLMConfig(
        provider=Provider.OPENAI,
        openai_api_key=SecretStr("private-openai-key"),
    )

    assert "private-openai-key" not in repr(config)
    assert "**********" in repr(config)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_config_rejects_temperature_outside_supported_range(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        LLMConfig(
            provider=Provider.OPENAI,
            openai_api_key=SecretStr("test-key"),
            temperature=temperature,
        )


def test_chat_request_requires_user_message() -> None:
    with pytest.raises(ValidationError, match="user message"):
        ChatRequest(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="System instruction")
            ]
        )


def test_stream_event_accepts_exactly_one_event_kind() -> None:
    assert StreamEvent(content="token").content == "token"
    assert StreamEvent(done=True).done is True

    with pytest.raises(ValidationError, match="content, an error, or done"):
        StreamEvent()

    with pytest.raises(ValidationError, match="content, an error, or done"):
        StreamEvent(content="token", done=True)
