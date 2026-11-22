"""Optional provider adapters; the scientific engine never imports this module."""

from __future__ import annotations

import os
from math import isfinite
from threading import Event
from typing import Any

from .contracts import (
    AssistantToolCall,
    AssistantToolSchema,
    ChatMessage,
    ProviderResponse,
)


class ProviderError(RuntimeError):
    """Base class for assistant transport errors with safe user-facing messages."""


class ProviderNotConfigured(ProviderError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderCancelled(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class OpenAIProvider:
    """OpenAI Responses API adapter using process-environment credentials only."""

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("NEUROSCOPE_OPENAI_MODEL")
        self.timeout_s = _positive_float(
            timeout_s
            if timeout_s is not None
            else os.environ.get("NEUROSCOPE_OPENAI_TIMEOUT_S", "35"),
            "NEUROSCOPE_OPENAI_TIMEOUT_S",
        )
        self.max_output_tokens = _positive_int(
            max_output_tokens
            if max_output_tokens is not None
            else os.environ.get("NEUROSCOPE_OPENAI_MAX_OUTPUT_TOKENS", "1200"),
            "NEUROSCOPE_OPENAI_MAX_OUTPUT_TOKENS",
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self.model)

    def respond(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[AssistantToolSchema, ...],
        *,
        timeout_s: float,
        cancel_event: Event | None = None,
    ) -> ProviderResponse:
        timeout = _positive_float(timeout_s, "timeout_s")
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderCancelled("Assistant request cancelled.")
        if not self._api_key or not self.model:
            raise ProviderNotConfigured(
                "Provider unavailable. Configure OPENAI_API_KEY and "
                "NEUROSCOPE_OPENAI_MODEL; scientific workflows remain available offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderNotConfigured(
                "The optional OpenAI client is not installed. Install the project's "
                "optional 'ai' extra to enable chat."
            ) from exc

        request_input = _to_response_input(messages)
        request_tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            }
            for tool in tools
        ]
        try:
            client = OpenAI(api_key=self._api_key, timeout=timeout, max_retries=0)
            response = client.responses.create(
                model=self.model,
                input=request_input,
                tools=request_tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except Exception as exc:  # noqa: BLE001 - SDK errors may include private request data
            error_name = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if error_name in {"APITimeoutError", "TimeoutError"}:
                raise ProviderTimeout("The assistant provider request timed out.") from None
            if error_name == "AuthenticationError" or status_code == 401:
                raise ProviderAuthenticationError(
                    "OpenAI authentication failed. Check the configured API credential."
                ) from None
            raise ProviderRequestError(
                f"The assistant provider request failed ({error_name})."
            ) from None
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderCancelled("Assistant request cancelled after provider response.")
        status = getattr(response, "status", "completed")
        if status != "completed":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details is not None else None
            if status == "incomplete" and reason == "max_output_tokens":
                raise ProviderRequestError(
                    "The assistant response reached its output-token limit."
                )
            raise ProviderRequestError("The assistant provider returned an incomplete response.")

        calls: list[AssistantToolCall] = []
        text_parts: list[str] = []
        for item in getattr(response, "output", ()):
            if getattr(item, "type", None) == "function_call":
                try:
                    calls.append(
                        AssistantToolCall(
                            call_id=item.call_id,
                            name=item.name,
                            arguments_json=item.arguments,
                        )
                    )
                except (AttributeError, ValueError, TypeError) as exc:
                    raise ProviderRequestError(
                        "The assistant provider returned a malformed function call."
                    ) from exc
            elif getattr(item, "type", None) == "message":
                for content in getattr(item, "content", ()):
                    if getattr(content, "type", None) == "output_text":
                        text = getattr(content, "text", None)
                        if isinstance(text, str):
                            text_parts.append(text)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            content="\n".join(text_parts),
            tool_calls=tuple(calls),
            model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            input_tokens=_optional_int(getattr(usage, "input_tokens", None)),
            output_tokens=_optional_int(getattr(usage, "output_tokens", None)),
            total_tokens=_optional_int(getattr(usage, "total_tokens", None)),
        )


def _to_response_input(messages: tuple[ChatMessage, ...]) -> list[dict[str, Any]]:
    """Translate local history to stateless Responses input items."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            if not message.tool_call_id:
                raise ProviderRequestError("A tool result is missing its call identifier.")
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            if message.content:
                result.append({"role": "assistant", "content": message.content})
            result.extend(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments_json,
                }
                for call in message.tool_calls
            )
            continue
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        result.append(item)
    return result


def _positive_float(value: float | str, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return normalized


def _positive_int(value: int | str, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if normalized <= 0 or str(normalized) != str(value).strip():
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
