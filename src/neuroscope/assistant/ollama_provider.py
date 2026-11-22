"""Optional Ollama provider using the local HTTP API and neutral assistant contracts."""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
from dataclasses import dataclass
from math import isfinite
from threading import Event
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from .contracts import (
    AssistantToolCall,
    AssistantToolSchema,
    ChatMessage,
    ProviderResponse,
)
from .providers import (
    ProviderCancelled,
    ProviderError,
    ProviderNotConfigured,
    ProviderRequestError,
    ProviderTimeout,
)

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


class OllamaModelUnavailable(ProviderError):
    """The configured model is not installed in the selected Ollama server."""


class OllamaModelIncompatible(ProviderError):
    """Ollama did not confirm that the selected model supports tool calls."""


@dataclass(frozen=True, slots=True)
class OllamaServiceStatus:
    state: str
    models: tuple[str, ...] = ()
    selected_model: str | None = None
    supports_tools: bool | None = None
    message: str = ""
    tool_capable_models: tuple[str, ...] = ()


class OllamaProvider:
    """Synchronous Ollama `/api/chat` adapter; orchestration remains provider-neutral."""

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout_s: float | str | None = None,
        max_output_tokens: int | str | None = None,
    ) -> None:
        self.host = _normalize_host(
            host or os.environ.get("NEUROSCOPE_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        )
        self.model = (model or os.environ.get("NEUROSCOPE_OLLAMA_MODEL") or "").strip()
        self.timeout_s = _positive_float(
            timeout_s
            if timeout_s is not None
            else os.environ.get("NEUROSCOPE_OLLAMA_TIMEOUT_S", "180"),
            "NEUROSCOPE_OLLAMA_TIMEOUT_S",
        )
        self.request_count = 0
        self.request_cost = 2  # capability probe plus one chat endpoint call
        self.max_output_tokens = _positive_int(
            max_output_tokens
            if max_output_tokens is not None
            else os.environ.get("NEUROSCOPE_OLLAMA_MAX_OUTPUT_TOKENS", "800"),
            "NEUROSCOPE_OLLAMA_MAX_OUTPUT_TOKENS",
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.model)

    @property
    def is_remote(self) -> bool:
        return not _is_loopback_host(urllib.parse.urlsplit(self.host).hostname or "")

    def list_models(self, *, timeout_s: float | None = None) -> tuple[str, ...]:
        payload = self._request_json(
            "GET", "/api/tags", timeout_s=timeout_s or self.timeout_s
        )
        models = payload.get("models")
        if not isinstance(models, list):
            raise ProviderRequestError("Ollama returned an invalid installed-model list.")
        names = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return tuple(dict.fromkeys(names))

    def model_supports_tools(
        self, model: str | None = None, *, timeout_s: float | None = None
    ) -> bool | None:
        selected = (model or self.model).strip()
        if not selected:
            raise ProviderNotConfigured("Select an installed Ollama model first.")
        payload = self._request_json(
            "POST",
            "/api/show",
            body={"model": selected},
            timeout_s=timeout_s or self.timeout_s,
            model_lookup=True,
        )
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, list):
            return None
        return "tools" in capabilities

    def inspect_service(self, *, timeout_s: float | None = None) -> OllamaServiceStatus:
        selected = self.model or None
        try:
            models = self.list_models(timeout_s=timeout_s)
        except ProviderTimeout:
            return OllamaServiceStatus("timeout", selected_model=selected, message="Ollama timed out.")
        except ProviderError:
            return OllamaServiceStatus("unavailable", selected_model=selected, message="Ollama is not installed or not running at the configured endpoint.")
        if not models:
            return OllamaServiceStatus("no_models", message="Ollama is running, but no models are installed.")
        if selected is None:
            capable: list[str] = []
            unknown = False
            for candidate in models:
                try:
                    support = self.model_supports_tools(candidate, timeout_s=timeout_s)
                except ProviderTimeout:
                    return OllamaServiceStatus("timeout", models, message="Ollama timed out while checking model capabilities.")
                except ProviderError:
                    support = None
                if support is True:
                    capable.append(candidate)
                elif support is None:
                    unknown = True
            if capable:
                return OllamaServiceStatus(
                    "model_unselected", models, message="Select an installed tool-capable model.",
                    tool_capable_models=tuple(capable),
                )
            if unknown:
                return OllamaServiceStatus(
                    "capability_unknown", models, message="Ollama could not confirm tool-calling capability for installed models."
                )
            return OllamaServiceStatus(
                "no_compatible_models", models, message="No installed model reports tool-calling support."
            )
        if selected not in models:
            return OllamaServiceStatus("model_unavailable", models, selected, message="The selected model is not installed on this Ollama server.")
        try:
            compatible = self.model_supports_tools(selected, timeout_s=timeout_s)
        except ProviderTimeout:
            return OllamaServiceStatus("timeout", models, selected, message="Ollama timed out while checking the selected model.")
        except ProviderError:
            return OllamaServiceStatus("model_unavailable", models, selected, message="Ollama could not inspect the selected model.")
        if compatible is True:
            return OllamaServiceStatus("ready", models, selected, True, "Model is available and reports tool-calling support.")
        if compatible is False:
            return OllamaServiceStatus("incompatible", models, selected, False, "The selected model does not report tool-calling support.")
        return OllamaServiceStatus("capability_unknown", models, selected, None, "Ollama did not report whether this model supports tool calls; requests are disabled.")

    def respond(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[AssistantToolSchema, ...],
        *,
        timeout_s: float,
        cancel_event: Event | None = None,
    ) -> ProviderResponse:
        timeout = _positive_float(timeout_s, "timeout_s")
        _check_cancel(cancel_event)
        if not self.model:
            raise ProviderNotConfigured(
                "Select an installed model in the Ollama provider settings."
            )
        supported = self.model_supports_tools(timeout_s=timeout)
        _check_cancel(cancel_event)
        if supported is not True:
            raise OllamaModelIncompatible(
                "Ollama did not confirm tool-calling support for the selected model. "
                "Choose a model whose Ollama capabilities include 'tools'."
            )
        request_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        payload = self._request_json(
            "POST",
            "/api/chat",
            body={
                "model": self.model,
                "messages": _to_ollama_messages(messages),
                "tools": request_tools,
                "stream": False,
                "options": {"num_predict": self.max_output_tokens},
            },
            timeout_s=timeout,
            cancel_event=cancel_event,
        )
        _check_cancel(cancel_event)
        if payload.get("done") is False:
            raise ProviderRequestError("Ollama returned an unfinished chat response.")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ProviderRequestError("Ollama returned a malformed chat response.")
        calls: list[AssistantToolCall] = []
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise ProviderRequestError("Ollama returned malformed tool calls.")
        for raw_call in raw_calls:
            try:
                function = raw_call["function"]
                name = function["name"]
                arguments = function.get("arguments", {})
                arguments_json = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                )
                calls.append(
                    AssistantToolCall(
                        call_id=f"ollama-{uuid4().hex}",
                        name=name,
                        arguments_json=arguments_json,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderRequestError(
                    "Ollama returned a malformed function call."
                ) from exc
        usage_in = _optional_int(payload.get("prompt_eval_count"))
        usage_out = _optional_int(payload.get("eval_count"))
        return ProviderResponse(
            content=message.get("content", "") if isinstance(message.get("content", ""), str) else "",
            tool_calls=tuple(calls),
            model=payload.get("model") if isinstance(payload.get("model"), str) else self.model,
            response_id=None,
            input_tokens=usage_in,
            output_tokens=usage_out,
            total_tokens=(usage_in + usage_out) if usage_in is not None and usage_out is not None else None,
        )

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        body: dict[str, object] | None = None,
        timeout_s: float,
        cancel_event: Event | None = None,
        model_lookup: bool = False,
    ) -> dict[str, Any]:
        _check_cancel(cancel_event)
        self.request_count += 1
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.host + endpoint,
            data=data,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if model_lookup and exc.code == 404:
                raise OllamaModelUnavailable("The selected Ollama model is not installed.") from None
            raise ProviderRequestError(f"Ollama request failed with HTTP {exc.code}.") from None
        except TimeoutError as exc:
            raise ProviderTimeout("The Ollama request timed out.") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeout("The Ollama request timed out.") from None
            raise ProviderNotConfigured(
                "Ollama is unavailable. Start Ollama manually and verify the configured endpoint."
            ) from None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderRequestError("Ollama returned invalid JSON.") from exc
        except OSError as exc:
            raise ProviderNotConfigured("Ollama is unavailable at the configured endpoint.") from exc
        if not isinstance(decoded, dict):
            raise ProviderRequestError("Ollama returned an invalid response object.")
        return decoded


def _to_ollama_messages(messages: tuple[ChatMessage, ...]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for message in messages:
        if message.role == "tool":
            output.append(
                {
                    "role": "tool",
                    "tool_name": message.name or "",
                    "content": message.content,
                }
            )
            continue
        item: dict[str, object] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            tool_calls = []
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.arguments_json)
                except json.JSONDecodeError:
                    arguments = {"_malformed_arguments_json": call.arguments_json}
                tool_calls.append(
                    {"function": {"name": call.name, "arguments": arguments}}
                )
            item["tool_calls"] = tool_calls
        output.append(item)
    return output


def _normalize_host(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama host must be an HTTP(S) origin without credentials or path.") from exc
    netloc = parsed.hostname
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{parsed.scheme}://{netloc}"


def _is_loopback_host(host: str) -> bool:
    if host.lower() in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProviderCancelled("Ollama request cancelled.")


def _positive_float(value: float | str, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _positive_int(value: int | str, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0 or str(result) != str(value).strip():
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
