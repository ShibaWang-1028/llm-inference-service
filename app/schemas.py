"""Request models for the OpenAI-compatible API.

Kept permissive (extra fields allowed) so the gateway stays compatible with the
full OpenAI chat API while still checking the essentials: a model name and a
non-empty list of messages.
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


class HealthResponse(BaseModel):
    status: str
