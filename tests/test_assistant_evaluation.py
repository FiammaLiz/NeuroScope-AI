"""Deterministic offline scientific-assistant scenario evaluation tests."""

from __future__ import annotations

import json

import pytest

from neuroscope.assistant.evaluation import (
    evaluation_cases,
    main,
    run_live_evaluation,
    run_offline_evaluation,
)


def test_offline_evaluation_executes_all_documented_cases_and_reports_categories(tmp_path):
    output = tmp_path / "assistant-eval.json"
    report = run_offline_evaluation(output)

    assert report["case_count"] == 25
    assert report["passed"] == 25
    assert report["failed"] == 0
    assert report["live_model_used"] is False
    assert {case["case_id"] for case in report["cases"]} == {
        case.case_id for case in evaluation_cases()
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_case_definitions_include_question_metadata_behavior_and_forbidden_actions():
    for case in evaluation_cases():
        assert case.question
        assert case.dataset_metadata
        assert case.available_operations
        assert case.expected_tool_behavior
        assert case.forbidden_actions
        assert case.pass_criteria
        assert case.failure_category


def test_live_evaluation_is_not_executed_without_environment_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEUROSCOPE_OPENAI_MODEL", raising=False)
    output = tmp_path / "live-eval.json"
    report = run_live_evaluation(
        output_path=output,
        max_requests=2,
        max_tokens=100,
        timeout_s=3.0,
    )
    assert report["status"] == "not_executed"
    assert report["live_model_used"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_cli_requires_explicit_billable_authorization_before_live_mode():
    with pytest.raises(SystemExit) as error:
        main(["--live"])
    assert error.value.code == 2


def test_local_evaluation_is_opt_in_and_remote_endpoints_require_explicit_consent(tmp_path, monkeypatch):
    from neuroscope.assistant.evaluation import run_local_model_evaluation
    from neuroscope.assistant.ollama_provider import OllamaProvider

    called = []
    monkeypatch.setattr(
        "neuroscope.assistant.ollama_provider.urlopen",
        lambda *args, **kwargs: called.append(args),
    )
    no_model = run_local_model_evaluation(
        output_path=tmp_path / "none.json",
        max_requests=2,
        max_tokens=100,
        timeout_s=1,
        model=None,
    )
    assert no_model["status"] == "not_executed"
    assert called == []

    remote = OllamaProvider(host="https://example.invalid", model="qwen3:4b")
    assert remote.is_remote
    report = run_local_model_evaluation(
        output_path=tmp_path / "remote.json",
        max_requests=2,
        max_tokens=100,
        timeout_s=1,
        host=remote.host,
        model=remote.model,
        authorize_remote=False,
    )
    assert report["status"] == "not_executed"
    assert "authorize-remote" in report["reason"]
    assert called == []


def test_cli_rejects_conflicting_or_unauthorized_provider_modes():
    from neuroscope.assistant.evaluation import main

    with pytest.raises(SystemExit) as conflict:
        main(["--live", "--local", "--authorize-billable"])
    assert conflict.value.code == 2
    with pytest.raises(SystemExit) as remote:
        main(["--authorize-remote"])
    assert remote.value.code == 2
