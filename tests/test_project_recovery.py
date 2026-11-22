import shutil
from pathlib import Path

import pytest
from pydantic import Field

from neuroscope.analysis.contracts import (
    OperationContext,
    OperationDefinition,
    OperationModel,
    OperationResult,
)
from neuroscope.analysis.operations import OperationRunner, restore_execution_record
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.domain import DatasetProfile
from neuroscope.memory.project_store import ProjectStore


class _Input(OperationModel):
    value: int = Field(strict=True)


class _Parameters(OperationModel):
    multiplier: int = Field(strict=True)


class _Output(OperationModel):
    value: int = Field(strict=True)


def _registered_test_operation():
    calls = []

    def execute(input_data, parameters, context):
        calls.append(input_data.value)
        return OperationResult(_Output(value=input_data.value * parameters.multiplier))

    definition = OperationDefinition(
        operation_id="neuroscope.test.recovery",
        name="Recovery fixture",
        description="Deterministic operation used to test historical restoration.",
        version="1.0.0",
        implementation_version="fixture-v1",
        input_model=_Input,
        parameters_model=_Parameters,
        output_model=_Output,
        execute=execute,
    )
    registry = OperationRegistry()
    registry.register(definition)
    return registry, definition, calls


def _saved_snapshot():
    registry, definition, calls = _registered_test_operation()
    runner = OperationRunner(registry)
    prepared = runner.prepare(
        definition.operation_id, definition.version, {"value": 7}, {"multiplier": 3},
        context=OperationContext(),
    )
    approved = runner.approve(prepared, approved_by="Researcher")
    record = runner.execute(approved)
    assert calls == [7]
    return registry, definition, record, calls


def test_new_project_manifest_identity_survives_reopen(tmp_path: Path):
    project_path = tmp_path / "project"
    created = ProjectStore(project_path)

    reopened = ProjectStore.open_existing(project_path)

    assert created.project_id is not None
    assert reopened.project_id == created.project_id
    assert reopened.legacy_project is False


def test_invalid_manifest_fails_preflight_without_mutating_project(tmp_path: Path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    manifest = project_path / "project.json"
    manifest.write_text('{"schema_version":99}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported project schema version"):
        ProjectStore.open_existing(project_path)

    assert manifest.read_text(encoding="utf-8") == '{"schema_version":99}'
    assert not (project_path / "artifacts").exists()


def test_truncated_history_fails_open_preflight_without_rewriting_it(tmp_path: Path):
    project_path = tmp_path / "project"
    store = ProjectStore(project_path)
    store.execution_history_path.write_text('[{"execution_id":', encoding="utf-8")

    with pytest.raises(ValueError):
        ProjectStore.open_existing(project_path)

    assert store.execution_history_path.read_text(encoding="utf-8") == '[{"execution_id":'


def test_legacy_project_opens_without_creating_missing_artifact_directories(tmp_path: Path):
    project_path = tmp_path / "legacy"
    project_path.mkdir()
    (project_path / "execution_history.json").write_text("[]", encoding="utf-8")

    store = ProjectStore.open_existing(project_path)

    assert store.legacy_project is True
    assert not (project_path / "artifacts").exists()
    assert store.load_execution_history() == []


def test_malformed_source_identity_is_reported_without_blocking_history_recovery(tmp_path: Path):
    project_path = tmp_path / "project"
    store = ProjectStore(project_path)
    store.source_identity_path.write_text("{broken", encoding="utf-8")

    reopened = ProjectStore.open_existing(project_path)

    assert reopened.source_identity_status == "invalid"
    assert reopened.load_execution_history() == []


def test_source_reconnection_uses_structure_and_partial_content_not_path_alone(tmp_path: Path):
    original = tmp_path / "original.bin"
    original.write_bytes(bytes(range(256)) * 600)
    profile = DatasetProfile(
        dataset_id="dataset-original", name="source", source_path=str(original),
        file_format="matlab-v7.3", dataset_path="recording/data", shape=[4, 1000], dtype="float32",
    )
    store = ProjectStore(tmp_path / "project")
    store.save_profile(profile)

    moved = tmp_path / "moved.bin"
    shutil.copyfile(original, moved)
    moved_profile = profile.model_copy(update={"source_path": str(moved)})
    comparison = store.source_match(moved_profile)
    assert comparison["status"] == "partial_match_confirmation_required"
    assert comparison["matches"] is True
    assert comparison["same_path"] is False

    changed = bytearray(moved.read_bytes())
    changed[0] ^= 1
    moved.write_bytes(changed)
    assert store.source_match(moved_profile)["status"] == "mismatch"


def test_additional_dataset_profile_does_not_replace_primary_source(tmp_path: Path):
    primary_path = tmp_path / "primary.npy"
    secondary_path = tmp_path / "secondary.npy"
    primary_path.write_bytes(b"primary source bytes")
    secondary_path.write_bytes(b"secondary source bytes")
    primary = DatasetProfile(
        dataset_id="primary", name="primary", source_path=str(primary_path),
        file_format="npy", shape=[3], dtype="float64",
    )
    secondary = primary.model_copy(
        update={"dataset_id": "secondary", "name": "secondary", "source_path": str(secondary_path)}
    )
    store = ProjectStore(tmp_path / "project")
    store.save_profile(primary)
    store.save_profile(secondary)

    assert store.load_profile().dataset_id == "primary"
    assert {item.dataset_id for item in store.load_profiles()} == {"primary", "secondary"}


def test_execution_snapshot_restores_as_history_without_running_operation(tmp_path: Path):
    registry, definition, record, calls = _saved_snapshot()
    calls_before_restore = list(calls)
    store = ProjectStore(tmp_path / "project")
    store.append_execution_record(record)

    recovered, rejected = store.restore_execution_records(registry)

    assert rejected == []
    assert len(recovered) == 1
    assert recovered[0].execution_id == record.execution_id
    assert recovered[0].output.value == 21
    assert recovered[0].provenance.approval_id == record.provenance.approval_id
    assert calls == calls_before_restore
    assert not hasattr(recovered[0], "_runner_token")
    assert definition.operation_id == "neuroscope.test.recovery"


def test_execution_restore_rejects_unsupported_implementation_version():
    _registry, definition, record, _calls = _saved_snapshot()
    snapshot = record.to_snapshot()
    snapshot["implementation_version"] = "newer-unknown-version"

    with pytest.raises(ValueError, match="implementation version is unsupported"):
        restore_execution_record(snapshot, definition)


def test_artifact_discovery_reports_incomplete_and_corrupt_entries(tmp_path: Path):
    store = ProjectStore(tmp_path / "project")
    incomplete = store.artifacts.root / ".pending-interrupted"
    incomplete.mkdir()
    corrupt = store.isi_artifacts.root / "00000000-0000-0000-0000-000000000000"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{bad", encoding="utf-8")

    entries = store.discover_artifacts()

    assert any(item["status"] == "incomplete_write" for item in entries)
    invalid = next(item for item in entries if item["artifact_id"].startswith("00000000"))
    assert invalid["status"] == "invalid"


def test_artifact_with_missing_execution_lineage_is_not_marked_available(tmp_path: Path):
    store = ProjectStore(tmp_path / "project")
    artifact_id = "00000000-0000-0000-0000-000000000001"
    directory = store.isi_artifacts.root / artifact_id
    directory.mkdir()
    (directory / "manifest.json").write_text(
        '{"artifact_id":"00000000-0000-0000-0000-000000000001",'
        '"intervals_file":"intervals.jsonl","interval_count":0}',
        encoding="utf-8",
    )
    (directory / "intervals.jsonl").write_text("", encoding="utf-8")

    entries = store.discover_artifacts()

    item = next(entry for entry in entries if entry["artifact_id"] == artifact_id)
    assert item["status"] == "lineage_unverified"


def test_artifact_discovery_reports_missing_payload_and_unsupported_schema(tmp_path: Path):
    store = ProjectStore(tmp_path / "project")
    missing_id = "00000000-0000-0000-0000-000000000002"
    missing = store.isi_artifacts.root / missing_id
    missing.mkdir()
    (missing / "manifest.json").write_text(
        '{"artifact_id":"00000000-0000-0000-0000-000000000002",'
        '"intervals_file":"intervals.jsonl","interval_count":0}',
        encoding="utf-8",
    )
    future_id = "00000000-0000-0000-0000-000000000003"
    future = store.artifacts.root / future_id
    future.mkdir()
    (future / "manifest.json").write_text(
        '{"artifact_id":"00000000-0000-0000-0000-000000000003",'
        '"schema_version":"9.0.0"}',
        encoding="utf-8",
    )

    entries = store.discover_artifacts()

    statuses = {item["artifact_id"]: item["status"] for item in entries}
    assert statuses[missing_id] == "invalid"
    assert statuses[future_id] == "unsupported_schema"
