import os
from collections.abc import AsyncIterator, Mapping
from types import TracebackType
from typing import Self

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import GenerationRequest, GenerationResult, Provider, StreamEvent


class AsyncLLMManager:
    """Routes generation requests to interchangeable asynchronous clients."""

    def __init__(self, clients: Mapping[Provider, BaseLLMClient]) -> None:
        if not clients:
            raise ValueError("At least one LLM client must be configured.")
        self._clients = dict(clients)

    @classmethod
    def from_environment(cls) -> Self:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        clients: dict[Provider, BaseLLMClient] = {}

        if openai_key := os.getenv("OPENAI_API_KEY"):
            clients[Provider.OPENAI] = OpenAIClient(
                api_key=openai_key,
                default_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            clients[Provider.ANTHROPIC] = AnthropicClient(
                api_key=anthropic_key,
                default_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if not clients:
            raise ValueError(
                "Configure OPENAI_API_KEY, ANTHROPIC_API_KEY, or both in the environment."
            )
        return cls(clients)

    def _client_for(self, provider: Provider) -> BaseLLMClient:
        try:
            return self._clients[provider]
        except KeyError as exc:
            raise ValueError(
                f"Provider '{provider.value}' is not configured."
            ) from exc

    async def generate(
        self,
        provider: Provider,
        request: GenerationRequest,
    ) -> GenerationResult:
        return await self._client_for(provider).generate(request)

    def stream(
        self,
        provider: Provider,
        request: GenerationRequest,
    ) -> AsyncIterator[StreamEvent]:
        return self._client_for(provider).stream(request)

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
