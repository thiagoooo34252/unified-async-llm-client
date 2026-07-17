from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ErrorType(StrEnum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    API = "api"
    EMPTY_RESPONSE = "empty_response"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: NonEmptyText


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def require_user_message(self) -> Self:
        if not any(message.role is MessageRole.USER for message in self.messages):
            raise ValueError("At least one user message is required")
        return self


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: NonEmptyText = "gpt-5.6-luna"
    anthropic_model: NonEmptyText = "claude-sonnet-5"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def require_provider_api_key(self) -> Self:
        if not self.api_key.get_secret_value().strip():
            raise ValueError(f"The {self.provider.value} API key cannot be blank")
        return self

    @property
    def api_key(self) -> SecretStr:
        key = (
            self.openai_api_key
            if self.provider is Provider.OPENAI
            else self.anthropic_api_key
        )
        if key is None:
            raise ValueError(f"The {self.provider.value} API key is not configured")
        return key

    @property
    def model(self) -> str:
        if self.provider is Provider.OPENAI:
            return self.openai_model
        return self.anthropic_model


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: NonEmptyText
    content: NonEmptyText
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: NonEmptyText
    error_type: ErrorType
    message: NonEmptyText
    retryable: bool


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str | None = None
    error: ErrorResponse | None = None
    done: bool = False

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        populated = sum(
            (
                bool(self.content),
                self.error is not None,
                self.done,
            )
        )
        if populated != 1:
            raise ValueError("A stream event must contain content, an error, or done")
        return self


GenerationResult = ModelResponse | ErrorResponse
