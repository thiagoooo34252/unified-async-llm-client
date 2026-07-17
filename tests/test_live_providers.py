import os

import pytest
from dotenv import load_dotenv
from pydantic import SecretStr

from manager import AsyncLLMManager
from schemas import (
    ChatMessage,
    ErrorResponse,
    LLMConfig,
    MessageRole,
    ModelResponse,
    Provider,
)

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_LLM_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_TESTS_ENABLED,
        reason="Set RUN_LIVE_LLM_TESTS=1 to call real provider APIs",
    ),
]


def live_config(provider: Provider) -> LLMConfig:
    load_dotenv()

    if provider is Provider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY is required for live tests")
        return LLMConfig(
            provider=provider,
            openai_api_key=SecretStr(api_key),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            max_tokens=64,
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.fail("ANTHROPIC_API_KEY is required for live tests")
    return LLMConfig(
        provider=provider,
        anthropic_api_key=SecretStr(api_key),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        max_tokens=64,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [Provider.OPENAI, Provider.ANTHROPIC],
    ids=lambda provider: provider.value,
)
async def test_live_provider_generates_and_streams(provider: Provider) -> None:
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="¿Qué es la entropía? Responde en una sola oración.",
        )
    ]

    async with AsyncLLMManager(live_config(provider)) as manager:
        response = await manager.generate(messages)
        events = [event async for event in manager.stream(messages)]

    if isinstance(response, ErrorResponse):
        pytest.fail(
            f"{provider.value} generation failed: "
            f"{response.error_type}: {response.message}"
        )

    assert isinstance(response, ModelResponse)
    assert response.content.strip()
    assert not [event.error for event in events if event.error]
    assert "".join(event.content or "" for event in events).strip()
    assert events[-1].done is True
