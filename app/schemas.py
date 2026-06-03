"""OpenAI-compatible request/response models.

These are kept permissive (extra fields allowed) so the gateway stays compatible
with the full OpenAI chat API while still validating the essentials: a model name
and a non-empty list of messages.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    # str for normal chat, list for multimodal/tool content; stay loose
    content: Any = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str
