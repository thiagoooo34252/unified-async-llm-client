from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ErrorCode(StrEnum):
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_error"
    TIMEOUT = "timeout_error"
    CONNECTION = "connection_error"
    PROVIDER = "provider_error"
    EMPTY_RESPONSE = "empty_response"
    CONFIGURATION = "configuration_error"


class StreamEventType(StrEnum):
    DELTA = "delta"
    COMPLETED = "completed"
    ERROR = "error"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: Annotated[str, Field(min_length=1, max_length=100_000)]
    model: Annotated[str | None, Field(min_length=1)] = None
    system_prompt: Annotated[str | None, Field(min_length=1, max_length=20_000)] = None
    max_tokens: Annotated[int, Field(ge=1, le=32_000)] = 1024
    temperature: Annotated[float, Field(ge=0, le=1)] = 0.2


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: Literal[True] = True
    provider: Provider
    model: str
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    @computed_field
    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: Literal[False] = False
    provider: Provider
    code: ErrorCode
    message: str
    retryable: bool = False


GenerationResult: TypeAlias = GenerationResponse | ErrorResponse


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str
    type: StreamEventType
    delta: str | None = None
    error: ErrorResponse | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "StreamEvent":
        if self.type is StreamEventType.DELTA:
            if self.delta is None:
                raise ValueError("Delta events require text.")
            if self.error is not None:
                raise ValueError("Delta events cannot include an error.")
        elif self.type is StreamEventType.ERROR:
            if self.error is None:
                raise ValueError("Error events require an error payload.")
            if self.delta is not None:
                raise ValueError("Error events cannot include text.")
        elif self.delta is not None or self.error is not None:
            raise ValueError("Completed events cannot include a payload.")
        return self
