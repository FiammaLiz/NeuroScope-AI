"""Provider-neutral, JSON-safe contracts for the scientific assistant."""

from __future__ import annotations

from threading import Event
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class AssistantModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(AssistantModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[AssistantToolCall, ...] = ()


class AssistantToolCall(AssistantModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class AssistantToolSchema(AssistantModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue]


class ProviderResponse(AssistantModel):
    content: str = ""
    tool_calls: tuple[AssistantToolCall, ...] = ()
    model: str | None = None
    response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class Provider(Protocol):
    """Model adapter. It knows nothing about NeuroScope's scientific runtime."""

    def respond(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[AssistantToolSchema, ...],
        *,
        timeout_s: float,
        cancel_event: Event | None = None,
    ) -> ProviderResponse: ...


class AssistantToolResult(AssistantModel):
    name: str
    ok: bool
    summary: str
    data: dict[str, JsonValue] = Field(default_factory=dict)


class AssistantProposal(AssistantModel):
    """Human-reviewable proposal; it is not an approval or execution record."""

    proposal_id: str
    operation_id: str
    operation_version: str
    implementation_version: str
    operation_name: str
    input_snapshot: dict[str, JsonValue]
    parameter_snapshot: dict[str, JsonValue]
    metadata_snapshot: tuple[dict[str, JsonValue], ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    state_generation: int
    status: Literal["prepared_not_executed"] = "prepared_not_executed"


class AssistantTurn(AssistantModel):
    """One provider turn plus local tool summaries and prepared proposals."""

    answer: str
    tool_results: tuple[AssistantToolResult, ...] = ()
    proposals: tuple[AssistantProposal, ...] = ()
    conversation: tuple[ChatMessage, ...] = ()
    model: str | None = None
    response_ids: tuple[str, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    profile_name: str | None = None


class GetExecutionSummaryArgs(AssistantModel):
    execution_id: str = Field(min_length=1, max_length=128)


class DescribeOperationArgs(AssistantModel):
    operation_id: str = Field(min_length=1, max_length=128)
    version: str | None = Field(default=None, max_length=64)


class PrepareOperationArgs(AssistantModel):
    operation_id: str = Field(min_length=1, max_length=128)
    operation_version: str = Field(min_length=1, max_length=64)
    input_data: dict[str, JsonValue]
    parameters: dict[str, JsonValue]


class SearchLocalLibraryArgs(AssistantModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=5)


class EmptyToolArgs(AssistantModel):
    pass

