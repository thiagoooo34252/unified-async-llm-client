import asyncio
import os

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ErrorResponse, GenerationRequest, Provider, StreamEventType

QUESTION = "¿Qué es la entropía?"


def _read_provider() -> Provider:
    configured = os.getenv("LLM_PROVIDER", Provider.OPENAI.value).strip().lower()
    value = input(f"Proveedor [openai/anthropic] ({configured}): ").strip().lower()
    return Provider(value or configured)


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value else default


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return float(raw_value) if raw_value else default


async def main() -> None:
    load_dotenv()

    provider = _read_provider()
    request = GenerationRequest(
        prompt=QUESTION,
        max_tokens=_read_int("LLM_MAX_TOKENS", 1024),
        temperature=_read_float("LLM_TEMPERATURE", 0.2),
    )

    async with AsyncLLMManager.from_environment() as manager:
        print("\nRespuesta completa:\n")
        result = await manager.generate(provider, request)
        if isinstance(result, ErrorResponse):
            print(f"[{result.code}] {result.message}")
        else:
            print(result.text)
            print(
                f"\nProveedor: {result.provider.value} | Modelo: {result.model} | "
                f"Tokens: {result.total_tokens or 'n/d'}"
            )

        print("\nStreaming:\n")
        async for event in manager.stream(provider, request):
            if event.type is StreamEventType.DELTA:
                print(event.delta, end="", flush=True)
            elif event.type is StreamEventType.ERROR and event.error:
                print(f"\n[{event.error.code}] {event.error.message}")
            elif event.type is StreamEventType.COMPLETED:
                print("\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Configuración inválida: {exc}") from exc
