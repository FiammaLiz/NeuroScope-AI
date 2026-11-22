"""Mocked Ollama transport tests; no server, model or network required."""

from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from neuroscope.assistant.contracts import AssistantToolSchema, ChatMessage
from neuroscope.assistant.ollama_provider import (
    OllamaModelIncompatible,
    OllamaProvider,
    OllamaServiceStatus,
)
from neuroscope.assistant.providers import (
    ProviderCancelled,
    ProviderNotConfigured,
    ProviderTimeout,
)


class _Response:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._data


def _tool():
    return AssistantToolSchema(
        name="get_metadata_status",
        description="Read metadata status.",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def test_provider_maps_ollama_chat_tool_calls_and_usage(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/api/show"):
            return _Response({"capabilities": ["completion", "tools"]})
        return _Response(
            {
                "model": "qwen3:4b",
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_metadata_status",
                                "arguments": {},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 23,
                "eval_count": 4,
            }
        )

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    provider = OllamaProvider(model="qwen3:4b")
    result = provider.respond(
        (
            ChatMessage(role="system", content="Follow local rules."),
            ChatMessage(role="user", content="Inspect metadata."),
        ),
        (_tool(),),
        timeout_s=17,
    )

    assert provider.is_configured
    assert not provider.is_remote
    assert len(requests) == 2
    assert result.model == "qwen3:4b"
    assert result.tool_calls[0].name == "get_metadata_status"
    assert json.loads(result.tool_calls[0].arguments_json) == {}
    assert result.input_tokens == 23
    assert result.output_tokens == 4
    assert result.total_tokens == 27
    chat_request = json.loads(requests[1][0].data)
    assert chat_request["stream"] is False
    assert chat_request["tools"][0]["function"]["name"] == "get_metadata_status"
    assert "api_key" not in chat_request


def test_tool_result_history_uses_ollama_tool_message_shape():
    from neuroscope.assistant.ollama_provider import _to_ollama_messages

    messages = _to_ollama_messages(
        (
            ChatMessage(role="assistant", content="", tool_calls=()),
            ChatMessage(role="tool", name="get_metadata_status", tool_call_id="local-id", content='{"ok":true}'),
        )
    )
    assert messages[0] == {"role": "assistant", "content": ""}
    assert messages[1] == {
        "role": "tool",
        "tool_name": "get_metadata_status",
        "content": '{"ok":true}',
    }


def test_provider_refuses_unconfirmed_or_missing_tool_capability(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response({"capabilities": ["completion"]})

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    provider = OllamaProvider(model="plain-model")
    with pytest.raises(OllamaModelIncompatible):
        provider.respond((ChatMessage(role="user", content="hello"),), (_tool(),), timeout_s=1)
    assert calls == ["http://127.0.0.1:11434/api/show"]

    monkeypatch.setattr(
        "neuroscope.assistant.ollama_provider.urlopen",
        lambda *_args, **_kwargs: _Response({}),
    )
    assert provider.model_supports_tools() is None
    with pytest.raises(OllamaModelIncompatible):
        provider.respond((ChatMessage(role="user", content="hello"),), (_tool(),), timeout_s=1)


def test_installed_model_discovery_and_service_status(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return _Response({"models": [{"name": "qwen3:4b"}, {"model": "qwen3:1.7b"}]})
        return _Response({"capabilities": ["tools"]})

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    provider = OllamaProvider(model="qwen3:4b")
    assert provider.list_models() == ("qwen3:4b", "qwen3:1.7b")
    assert provider.inspect_service() == OllamaServiceStatus(
        "ready", ("qwen3:4b", "qwen3:1.7b"), "qwen3:4b", True,
        "Model is available and reports tool-calling support.",
    )


def test_service_status_handles_empty_models_and_unavailable_service(monkeypatch):
    monkeypatch.setattr(
        "neuroscope.assistant.ollama_provider.urlopen",
        lambda *_args, **_kwargs: _Response({"models": []}),
    )
    assert OllamaProvider().inspect_service().state == "no_models"

    def unavailable(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", unavailable)
    assert OllamaProvider().inspect_service().state == "unavailable"


def test_loopback_remote_classification_and_host_validation():
    assert not OllamaProvider(host="http://127.0.0.1:11434").is_remote
    assert not OllamaProvider(host="http://localhost:11434").is_remote
    assert OllamaProvider(host="http://192.0.2.4:11434").is_remote
    with pytest.raises(ValueError):
        OllamaProvider(host="http://user:password@localhost:11434")
    with pytest.raises(ValueError):
        OllamaProvider(host="http://localhost:11434/unrestricted-path")


def test_pre_cancel_and_provider_timeout_are_classified(monkeypatch):
    from threading import Event

    cancelled = Event()
    cancelled.set()
    with pytest.raises(ProviderCancelled):
        OllamaProvider(model="qwen3:4b").respond((), (), timeout_s=1, cancel_event=cancelled)

    def timeout(*_args, **_kwargs):
        raise TimeoutError("transport timeout")

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", timeout)
    with pytest.raises(ProviderTimeout):
        OllamaProvider(model="qwen3:4b").list_models(timeout_s=1)


def test_missing_model_does_not_make_a_request(monkeypatch):
    called = False

    def should_not_call(*_args, **_kwargs):
        nonlocal called
        called = True
        return _Response({})

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", should_not_call)
    with pytest.raises(ProviderNotConfigured):
        OllamaProvider().respond((), (), timeout_s=1)
    assert not called


def test_orchestrator_uses_shared_local_tool_allowlist_and_returns_tool_result(monkeypatch, tmp_path):
    from neuroscope.assistant.evaluation import _build_synthetic_state
    from neuroscope.assistant.orchestration import AssistantOrchestrator
    from neuroscope.assistant.tools import AssistantToolService

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/api/show"):
            return _Response({"capabilities": ["tools"]})
        body = json.loads(request.data)
        if len([message for message in body["messages"] if message["role"] == "tool"]) == 0:
            return _Response(
                {
                    "model": "qwen3:4b",
                    "done": True,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_metadata_status",
                                    "arguments": {},
                                }
                            }
                        ],
                    },
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                }
            )
        assert body["messages"][-1]["tool_name"] == "get_metadata_status"
        return _Response(
            {
                "model": "qwen3:4b",
                "done": True,
                "message": {"role": "assistant", "content": "Sampling rate is confirmed at 1000 Hz."},
                "prompt_eval_count": 20,
                "eval_count": 8,
            }
        )

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    state = _build_synthetic_state(tmp_path)
    provider = OllamaProvider(model="qwen3:4b")
    turn = AssistantOrchestrator(provider, AssistantToolService(lambda: state)).respond(
        (ChatMessage(role="user", content="What is the sampling rate?"),), timeout_s=3
    )

    assert turn.answer == "Sampling rate is confirmed at 1000 Hz."
    assert len(turn.tool_results) == 1 and turn.tool_results[0].ok
    assert turn.tool_results[0].data["metadata"]["sampling_rate_hz"] == {
        "value": 1000.0,
        "status": "confirmed",
    }
    chat_payloads = [json.loads(request.data) for request in requests if request.full_url.endswith("/api/chat")]
    assert len(chat_payloads) == 2
    assert chat_payloads[1]["messages"][-1]["role"] == "tool"
    assert chat_payloads[1]["messages"][-1]["tool_name"] == "get_metadata_status"



def test_service_reports_no_compatible_models_when_none_declares_tools(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/tags"):
            return _Response({"models": [{"name": "text-only"}]})
        return _Response({"capabilities": ["completion"]})

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    status = OllamaProvider().inspect_service()
    assert status.state == "no_compatible_models"
    assert status.models == ("text-only",)
    assert status.tool_capable_models == ()
