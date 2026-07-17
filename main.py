import asyncio
import os

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from manager import AsyncLLMManager
from schemas import ChatMessage, ErrorResponse, LLMConfig, MessageRole, Provider


def load_config() -> LLMConfig:
    load_dotenv()
    provider_name = os.getenv("LLM_PROVIDER", Provider.OPENAI.value)
    try:
        provider = Provider(provider_name)
    except ValueError as exc:
        supported = ", ".join(provider.value for provider in Provider)
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. Use: {supported}"
        ) from exc

    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    return LLMConfig(
        provider=provider,
        openai_api_key=SecretStr(openai_api_key) if openai_api_key else None,
        anthropic_api_key=(SecretStr(anthropic_api_key) if anthropic_api_key else None),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


async def run() -> None:
    try:
        config = load_config()
    except (ValidationError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Respond clearly and in no more than two sentences.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="¿Qué es la entropía?",
        ),
    ]

    async with AsyncLLMManager(config) as manager:
        print("Normal response:")
        response = await manager.generate(messages)
        if isinstance(response, ErrorResponse):
            print(f"[{response.error_type}] {response.message}")
            return
        print(response.content)

        print("\nStreaming response:")
        async for event in manager.stream(messages):
            if event.error:
                print(f"\n[{event.error.error_type}] {event.error.message}")
                return
            if event.content:
                print(event.content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(run())
