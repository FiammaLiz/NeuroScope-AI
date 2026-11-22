"""OpenAI Responses adapter tests without live requests or credentials."""

from __future__ import annotations

import sys
from threading import Event
from types import SimpleNamespace

import pytest

from neuroscope.assistant.contracts import (
    AssistantToolCall,
    AssistantToolSchema,
    ChatMessage,
)
from neuroscope.assistant.providers import (
    OpenAIProvider,
    ProviderAuthenticationError,
    ProviderCancelled,
    ProviderNotConfigured,
    ProviderRequestError,
    ProviderTimeout,
)


def _function_call(call_id="call-a", name="list_available_operations", arguments="{}"):
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def _text_message(text):
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def test_responses_adapter_maps_calls_tools_history_usage_and_store_policy(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        id="resp-1",
        model="documented-test-model",
        status="completed",
        output=[_function_call()],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
    )

    class FakeOpenAI:
        def __init__(self, *, api_key, timeout, max_retries):
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            captured["request"] = kwargs
            return response

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-in-request")
    monkeypatch.setenv("NEUROSCOPE_OPENAI_MODEL", "documented-test-model")
    monkeypatch.setenv("NEUROSCOPE_OPENAI_TIMEOUT_S", "8")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    provider = OpenAIProvider()
    tool = AssistantToolSchema(
        name="list_available_operations",
        description="List operations",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    messages = (
        ChatMessage(role="system", content="guardrail"),
        ChatMessage(role="user", content="inspect"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=(AssistantToolCall(call_id="call-old", name="inspect", arguments_json="{}"),),
        ),
        ChatMessage(role="tool", name="inspect", tool_call_id="call-old", content='{"ok":true}'),
    )
    result = provider.respond(messages, (tool,), timeout_s=8.0)

    assert provider.is_configured
    assert result.tool_calls == (AssistantToolCall(call_id="call-a", name="list_available_operations", arguments_json="{}"),)
    assert result.model == "documented-test-model"
    assert result.response_id == "resp-1"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (12, 3, 15)
    assert captured["timeout"] == 8.0
    assert captured["max_retries"] == 0
    request = captured["request"]
    assert request["store"] is False
    assert request["parallel_tool_calls"] is False
    assert request["tools"] == [
        {
            "type": "function",
            "name": "list_available_operations",
            "description": "List operations",
            "parameters": tool.parameters,
            "strict": False,
        }
    ]
    assert request["input"][-2:] == [
        {"type": "function_call", "call_id": "call-old", "name": "inspect", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-old", "output": '{"ok":true}'},
    ]
    assert "test-secret-never-in-request" not in repr(request)


def test_responses_adapter_extracts_final_text_and_rejects_incomplete(monkeypatch):
    response = SimpleNamespace(
        id="resp-final",
        model="m",
        status="completed",
        output=[_text_message("Grounded answer.")],
        usage=None,
    )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(create=lambda **_request: response)

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("NEUROSCOPE_OPENAI_MODEL", "m")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    result = OpenAIProvider().respond((), (), timeout_s=2)
    assert result.content == "Grounded answer."

    response.status = "incomplete"
    response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    with pytest.raises(ProviderRequestError, match="output-token limit"):
        OpenAIProvider().respond((), (), timeout_s=2)


@pytest.mark.parametrize(
    ("error_name", "status", "expected"),
    [
        ("APITimeoutError", None, ProviderTimeout),
        ("AuthenticationError", 401, ProviderAuthenticationError),
        ("RuntimeError", 500, ProviderRequestError),
    ],
)
def test_responses_adapter_classifies_provider_errors_without_leaking_details(
    monkeypatch, error_name, status, expected
):
    error_type = type(error_name, (Exception,), {"status_code": status})

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(create=lambda **_request: self.fail())

        @staticmethod
        def fail():
            raise error_type("test-secret and private prompt")

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("NEUROSCOPE_OPENAI_MODEL", "m")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    with pytest.raises(expected) as captured:
        OpenAIProvider().respond((), (), timeout_s=1.0)
    assert "test-secret" not in str(captured.value)
    assert "private prompt" not in str(captured.value)


def test_configuration_limits_offline_and_pre_cancelled_requests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEUROSCOPE_OPENAI_MODEL", raising=False)
    with pytest.raises(ProviderNotConfigured):
        OpenAIProvider().respond((), (), timeout_s=1.0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("NEUROSCOPE_OPENAI_MODEL", "m")
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ProviderCancelled):
        OpenAIProvider().respond((), (), timeout_s=1.0, cancel_event=cancelled)

    monkeypatch.setenv("NEUROSCOPE_OPENAI_TIMEOUT_S", "NaN")
    with pytest.raises(ValueError, match="finite positive"):
        OpenAIProvider()
