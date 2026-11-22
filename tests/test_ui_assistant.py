"""Headless tests for consent, asynchronous chat, and trusted proposal approval."""

from __future__ import annotations

import os
import threading
import time
from threading import Event
from typing import ClassVar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from neuroscope.analysis.contracts import (
    OperationDefinition,
    OperationModel,
    OperationResult,
)
from neuroscope.analysis.operations import OperationRunner
from neuroscope.assistant.contracts import AssistantToolCall, ProviderResponse
from neuroscope.assistant.profiles import (
    DEFAULT_PROFILE_ID,
    GENERAL_BIOLOGICAL_SIGNALS_ID,
)
from neuroscope.assistant.providers import ProviderCancelled, ProviderTimeout
from neuroscope.memory.project_store import ProjectStore
from neuroscope.ui import main_window as main_window_module
from neuroscope.ui.main_window import MainWindow


class UIInput(OperationModel):
    count: int


class UIParameters(OperationModel):
    factor: int


class UIOutput(OperationModel):
    result: int


class FakeProvider:
    def __init__(self, responses, *, configured=True):
        self.responses = list(responses)
        self.configured = configured
        self.thread_ids = []

    @property
    def is_configured(self):
        return self.configured

    def respond(self, _messages, _tools, *, timeout_s, cancel_event=None):
        self.thread_ids.append(threading.get_ident())
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def _wait(app, condition, timeout=8.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert condition(), "assistant or operation worker did not finish"


def _select_project(window, tmp_path):
    window.store = ProjectStore(tmp_path / "project")
    window.operation_runner = OperationRunner(
        window.operation_registry,
        data_store=window.dataset_store,
        artifact_store=window.store.artifacts,
        isi_artifact_store=window.store.isi_artifacts,
        mua_artifact_store=window.store.mua_artifacts,
    )


def _consent(monkeypatch):
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Researcher A", True),
    )


def test_chat_runs_off_gui_thread_and_proposal_requires_trusted_researcher_approval(
    app, tmp_path, monkeypatch
):
    calls = []

    def execute(input_data, parameters, _context):
        calls.append((input_data.count, parameters.factor, threading.get_ident()))
        return OperationResult(UIOutput(result=input_data.count * parameters.factor))

    window = MainWindow()
    _select_project(window, tmp_path)
    window.operation_registry.register(
        OperationDefinition(
            operation_id="neuroscope.test.assistant-ui",
            name="GUI handoff test",
            description="A test operation with no scientific meaning.",
            version="1.0.0",
            implementation_version="ui-test-v1",
            input_model=UIInput,
            parameters_model=UIParameters,
            output_model=UIOutput,
            execute=execute,
        )
    )
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=(
                    AssistantToolCall(
                        call_id="proposal-call",
                        name="prepare_scientific_operation",
                        arguments_json=(
                            '{"operation_id":"neuroscope.test.assistant-ui",'
                            '"operation_version":"1.0.0",'
                            '"input_data":{"count":6},"parameters":{"factor":7}}'
                        ),
                    ),
                )
            ),
            ProviderResponse(content="I prepared the request for your review."),
        ]
    )
    monkeypatch.setattr(main_window_module, "OpenAIProvider", lambda: provider)
    # The window already owns a provider, so replace its instance and status explicitly.
    window._assistant_provider = provider
    window._update_assistant_provider_status()
    _consent(monkeypatch)
    main_thread_id = threading.get_ident()
    window.chat_input.setText("Prepare this simple operation.")
    window.send_chat()
    _wait(
        app,
        lambda: window._assistant_thread is None
        and window.assistant_proposal_list.count() == 1,
    )
    assert provider.thread_ids and provider.thread_ids[0] != main_thread_id
    assert calls == []
    proposal_text = window.assistant_proposal_details.toPlainText()
    assert "prepared only" in proposal_text.lower()
    assert "factor" in proposal_text

    window.assistant_approve_button.click()
    _wait(app, lambda: not window._execution_in_flight)
    assert calls and calls[0][:2] == (6, 7)
    assert calls[0][2] != main_thread_id
    history = window.store.load_execution_history()
    assert len(history) == 1
    assert history[0]["operation_id"] == "neuroscope.test.assistant-ui"
    assert history[0]["provenance"]["approved_by"] == "Researcher A"
    assert "Approved proposal outcome" in window.chat_history.toPlainText()
    window.close()
    app.processEvents()


def test_provider_failure_can_be_retried_and_offline_mode_sends_nothing(
    app, tmp_path, monkeypatch
):
    window = MainWindow()
    provider = FakeProvider(
        [ProviderTimeout("timeout"), ProviderResponse(content="Retry succeeded.")]
    )
    window._assistant_provider = provider
    _consent(monkeypatch)
    window.chat_input.setText("Explain unknown sampling rate.")
    window.send_chat()
    _wait(app, lambda: window._assistant_thread is None and window.assistant_retry_button.isEnabled())
    assert "timeout" in window.assistant_activity_label.text()
    window.retry_assistant_request()
    _wait(app, lambda: window._assistant_thread is None and "Retry succeeded." in window.chat_history.toPlainText())
    assert len(provider.thread_ids) == 2
    window.close()
    app.processEvents()

    offline = MainWindow()
    offline_provider = FakeProvider([], configured=False)
    offline._assistant_provider = offline_provider
    offline._update_assistant_provider_status()
    offline.chat_input.setText("Run local science.")
    offline.send_chat()
    assert not offline_provider.thread_ids
    assert "Offline" in offline.assistant_status_label.text()
    offline.close()
    app.processEvents()


def test_declined_privacy_consent_prevents_provider_request(app, monkeypatch):
    window = MainWindow()
    provider = FakeProvider([ProviderResponse(content="must not run")])
    window._assistant_provider = provider
    prompts = []

    def deny(*args, **kwargs):
        prompts.append(str(args[2]))
        return QMessageBox.No

    monkeypatch.setattr(main_window_module.QMessageBox, "question", deny)
    window.chat_input.setText("Inspect the dataset.")
    window.send_chat()
    assert not provider.thread_ids
    assert "metadata" in prompts[0]
    assert "Raw signal samples" in prompts[0]
    assert "Request not sent" in window.assistant_activity_label.text()
    window.close()
    app.processEvents()


def test_cancel_button_signals_provider_worker(app, monkeypatch):
    started = Event()

    class CancellableProvider(FakeProvider):
        def respond(self, messages, tools, *, timeout_s, cancel_event=None):
            self.thread_ids.append(threading.get_ident())
            started.set()
            deadline = time.monotonic() + 2.0
            while cancel_event is not None and not cancel_event.is_set() and time.monotonic() < deadline:
                time.sleep(0.005)
            if cancel_event is not None and cancel_event.is_set():
                raise ProviderCancelled("Assistant request cancelled.")
            return ProviderResponse(content="Cancellation was not received.")

    window = MainWindow()
    provider = CancellableProvider([])
    window._assistant_provider = provider
    _consent(monkeypatch)
    window.chat_input.setText("Explain the data metadata.")
    window.send_chat()
    assert started.wait(timeout=3)
    window.cancel_assistant_request()
    _wait(app, lambda: window._assistant_thread is None)
    assert "cancelled" in window.assistant_activity_label.text().lower()
    window.close()
    app.processEvents()


def test_provider_selector_uses_local_ollama_without_openai_fallback(app, monkeypatch):
    class MemorySettings:
        values: ClassVar[dict[str, str]] = {}

        def __init__(self, *_args):
            pass

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    MemorySettings.values = {}
    monkeypatch.setattr(main_window_module, "QSettings", MemorySettings)
    window = MainWindow()
    window.assistant_provider_box.setCurrentIndex(
        window.assistant_provider_box.findData("ollama")
    )
    assert isinstance(window._assistant_provider, main_window_module.OllamaProvider)
    assert not window._assistant_provider.is_configured
    assert "Ollama selected" in window.assistant_status_label.text()

    window.ollama_model_box.addItem("qwen3:4b", "qwen3:4b")
    window.ollama_model_box.setCurrentIndex(0)
    assert window._assistant_provider.model == "qwen3:4b"
    assert window._assistant_provider_kind == "ollama"
    assert MemorySettings.values["assistant/provider"] == "ollama"
    assert MemorySettings.values["assistant/ollama_model"] == "qwen3:4b"
    assert window._assistant_provider.host == "http://127.0.0.1:11434"
    window.close()
    app.processEvents()


def test_scientific_profile_selector_persists_without_provider_restart_or_consent(app, monkeypatch):
    class MemorySettings:
        values: ClassVar[dict[str, str]] = {}

        def __init__(self, *_args):
            pass

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    MemorySettings.values = {}
    monkeypatch.setattr(main_window_module, "QSettings", MemorySettings)
    window = MainWindow()
    initial_provider = window._assistant_provider
    assert window._assistant_scientific_profile.profile_id == DEFAULT_PROFILE_ID
    assert window.assistant_evidence_status.text().endswith("not connected")
    window.assistant_profile_box.setCurrentIndex(
        window.assistant_profile_box.findData(GENERAL_BIOLOGICAL_SIGNALS_ID)
    )
    assert window._assistant_scientific_profile.profile_id == GENERAL_BIOLOGICAL_SIGNALS_ID
    assert MemorySettings.values["assistant/scientific_profile"] == GENERAL_BIOLOGICAL_SIGNALS_ID
    assert window._assistant_provider is initial_provider
    assert "not model weights" in window.assistant_profile_details.text()
    window.close()
    app.processEvents()

    restored = MainWindow()
    assert restored._assistant_scientific_profile.profile_id == GENERAL_BIOLOGICAL_SIGNALS_ID
    restored.close()
    app.processEvents()



def test_remote_ollama_requires_endpoint_disclosure_before_chat(app, monkeypatch):
    class MemorySettings:
        values: ClassVar[dict[str, str]] = {}

        def __init__(self, *_args):
            pass

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    MemorySettings.values = {}
    monkeypatch.setattr(main_window_module, "QSettings", MemorySettings)
    window = MainWindow()
    window.ollama_host_input.setText("https://ollama.example.invalid")
    window._on_ollama_host_changed()
    window.ollama_model_box.addItem("qwen3:4b", "qwen3:4b")
    window.ollama_model_box.setCurrentIndex(0)
    window.assistant_provider_box.setCurrentIndex(
        window.assistant_provider_box.findData("ollama")
    )
    prompts = []

    def deny(*args, **kwargs):
        prompts.append(str(args[2]))
        return QMessageBox.No

    monkeypatch.setattr(main_window_module.QMessageBox, "question", deny)
    window.chat_input.setText("Inspect this synthetic project.")
    window.send_chat()
    assert prompts
    assert "non-loopback Ollama endpoint" in prompts[0]
    assert window._assistant_thread is None
    assert "Request not sent" in window.assistant_activity_label.text()
    window.close()
    app.processEvents()


def test_ollama_model_discovery_runs_on_worker_and_updates_ui(app, monkeypatch):
    import json
    import threading

    from neuroscope.assistant.ollama_provider import OllamaProvider

    class MemorySettings:
        values: ClassVar[dict[str, str]] = {}

        def __init__(self, *_args):
            pass

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    class FakeResponse:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.payload

    MemorySettings.values = {}
    monkeypatch.setattr(main_window_module, "QSettings", MemorySettings)
    thread_ids = []

    def fake_urlopen(request, timeout):
        thread_ids.append(threading.get_ident())
        if request.full_url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": "qwen3:4b"}]})
        return FakeResponse({"capabilities": ["completion", "tools"]})

    monkeypatch.setattr("neuroscope.assistant.ollama_provider.urlopen", fake_urlopen)
    window = MainWindow()
    window.assistant_provider_box.setCurrentIndex(
        window.assistant_provider_box.findData("ollama")
    )
    window.ollama_model_box.addItem("qwen3:4b", "qwen3:4b")
    window.ollama_model_box.setCurrentIndex(0)
    gui_thread = threading.get_ident()
    window.refresh_ollama_models()
    _wait(app, lambda: window._ollama_status_thread is None)

    assert len(thread_ids) == 2
    assert all(thread_id != gui_thread for thread_id in thread_ids)
    assert "ready" in window.assistant_status_label.text()
    assert window.ollama_model_box.currentData() == "qwen3:4b"
    assert isinstance(window._assistant_provider, OllamaProvider)
    window.close()
    app.processEvents()
