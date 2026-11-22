from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..analysis.operations import ExecutionRecord
from ..artifacts import ISIArtifactStore, WaveformArtifactStore
from ..domain import DatasetProfile
from ..mua_artifacts import MUAArtifactStore


class ProjectManifest(BaseModel):
    """Small versioned identity record for a NeuroScope project."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    project_id: UUID
    created_at_utc: str
    updated_at_utc: str


def _atomic_json(path: Path, payload: object, *, prefix: str) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = path.parent / f".{prefix}-{uuid4()}.tmp"
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _partial_file_fingerprint(path: Path, block_size: int = 65536) -> str:
    """Hash bounded first/middle/last file blocks; this is explicitly partial."""
    size = path.stat().st_size
    offsets = sorted({0, max(0, (size - block_size) // 2), max(0, size - block_size)})
    digest = hashlib.sha256()
    digest.update(f"size:{size};blocks:{block_size};".encode())
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            digest.update(f"offset:{offset};".encode())
            digest.update(stream.read(block_size))
    return digest.hexdigest()


class ProjectStore:
    """Local scientific project store with explicit create/open semantics."""

    MANIFEST_NAME = "project.json"
    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path, *, create: bool = True):
        self.root = Path(root)
        root_existed = self.root.exists()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise FileNotFoundError(f"Project directory does not exist: {self.root}")
        self.manifest_path = self.root / self.MANIFEST_NAME
        self.profile_path = self.root / "dataset_profile.json"
        self.profiles_directory = self.root / "dataset_profiles"
        self.source_identity_path = self.root / "dataset_source_identity.json"
        self.source_resolution_path = self.root / "dataset_source_resolution.json"
        self.history_path = self.root / "analysis_history.json"
        self.execution_history_path = self.root / "execution_history.json"
        self.pipelines_directory = self.root / "pipelines"
        # Literature remains project-scoped by default; no corpus is shared with
        # another project unless a researcher explicitly imports its manifest.
        from ..assistant.library import LiteratureLibraryStore

        self.literature = LiteratureLibraryStore(
            self.root / "literature", create=create
        )
        from ..assistant.knowledge_packs import KnowledgePackService

        self.knowledge_packs = KnowledgePackService(self.literature)
        self.literature_recovery_status = "not_checked"
        self.artifacts = WaveformArtifactStore(
            self.root / "artifacts" / "waveforms", create_root=create
        )
        self.isi_artifacts = ISIArtifactStore(
            self.root / "artifacts" / "isi", create_root=create
        )
        self.mua_artifacts = MUAArtifactStore(
            self.root / "artifacts" / "mua", create_root=create
        )
        self.manifest: ProjectManifest | None = None
        self.legacy_project = False
        self.source_identity_status = "missing"
        if create and not self.manifest_path.exists() and root_existed and any(self.root.iterdir()):
            self.legacy_project = True
        elif create and not self.manifest_path.exists():
            now = datetime.now(timezone.utc).isoformat()
            self.manifest = ProjectManifest(
                schema_version=self.SCHEMA_VERSION,
                project_id=uuid4(),
                created_at_utc=now,
                updated_at_utc=now,
            )
            self._save_manifest()
        elif self.manifest_path.exists():
            self.manifest = self._read_manifest()
        else:
            self.legacy_project = True
        if self.source_identity_path.exists():
            try:
                self.load_source_identity()
                self.source_identity_status = "valid"
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self.source_identity_status = "invalid"

    @classmethod
    def open_existing(cls, root: str | Path) -> ProjectStore:
        """Open without creating project files; legacy projects remain read-only until saved."""
        store = cls(root, create=False)
        # Validate persisted top-level records before a caller swaps its active project.
        store.load_profiles()
        store.load_history()
        store.load_execution_history()
        store.discover_artifacts()
        store.list_pipelines()
        store.list_pipeline_runs()
        try:
            store.literature.validate_existing()
            for library in store.literature.list_libraries():
                if library.knowledge_pack_id is not None and library.knowledge_pack_version is not None:
                    store.knowledge_packs.installed_manifest(
                        library.knowledge_pack_id,
                        library.knowledge_pack_version,
                        verify_payloads=False,
                    )
            store.literature_recovery_status = "validated"
        except (OSError, TypeError, ValueError) as exc:
            # A damaged optional literature catalog must not hide project analyses.
            store.literature_recovery_status = f"unavailable:{type(exc).__name__}"
        return store

    def _read_manifest(self) -> ProjectManifest:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Project manifest is invalid: {exc}") from exc
        if isinstance(raw, dict) and raw.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project schema version {raw.get('schema_version')!r}; "
                f"this application supports {self.SCHEMA_VERSION}."
            )
        try:
            manifest = ProjectManifest.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"Project manifest is invalid: {exc}") from exc
        if manifest.schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project schema version {manifest.schema_version}; "
                f"this application supports {self.SCHEMA_VERSION}."
            )
        return manifest

    def _save_manifest(self) -> None:
        if self.manifest is None:
            return
        _atomic_json(self.manifest_path, self.manifest.model_dump(mode="json"), prefix="project")

    @property
    def project_id(self) -> str | None:
        return str(self.manifest.project_id) if self.manifest is not None else None

    def save_profile(self, profile: DatasetProfile):
        existing = self.load_profile()
        if existing is not None and existing.dataset_id != profile.dataset_id:
            self.profiles_directory.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(profile.dataset_id.encode("utf-8")).hexdigest()
            destination = self.profiles_directory / f"{key}.json"
            _atomic_json(destination, profile.model_dump(mode="json"), prefix="dataset-profile")
            self._touch_manifest()
            return
        _atomic_json(self.profile_path, profile.model_dump(mode="json"), prefix="profile")
        source_path = Path(profile.source_path)
        if source_path.is_file():
            stat = source_path.stat()
            descriptor = {
                "dataset_id": profile.dataset_id,
                "source_path": str(source_path.resolve()),
                "file_format": profile.file_format,
                "dataset_path": profile.dataset_path,
                "shape": profile.shape,
                "dtype": profile.dtype,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "partial_fingerprint": _partial_file_fingerprint(source_path),
                "fingerprint_kind": "sha256_first_middle_last_blocks_not_whole_file",
            }
            _atomic_json(self.source_identity_path, descriptor, prefix="source-identity")
            self.source_identity_status = "valid"
        self._touch_manifest()

    def _touch_manifest(self) -> None:
        if self.manifest is None:
            return
        self.manifest = self.manifest.model_copy(
            update={"updated_at_utc": datetime.now(timezone.utc).isoformat()}
        )
        self._save_manifest()

    def load_profile(self) -> DatasetProfile | None:
        if not self.profile_path.exists():
            return None
        return DatasetProfile.model_validate_json(self.profile_path.read_text(encoding="utf-8"))

    def load_profiles(self) -> list[DatasetProfile]:
        profiles = []
        primary = self.load_profile()
        if primary is not None:
            profiles.append(primary)
        if self.profiles_directory.is_dir():
            for path in sorted(self.profiles_directory.glob("*.json")):
                profiles.append(
                    DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
                )
        return profiles

    def load_source_identity(self) -> dict[str, Any] | None:
        if not self.source_identity_path.is_file():
            return None
        value = json.loads(self.source_identity_path.read_text(encoding="utf-8"))
        required = {
            "dataset_id", "source_path", "file_format", "shape", "dtype",
            "size_bytes", "mtime_ns", "partial_fingerprint", "fingerprint_kind",
        }
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError("Saved dataset source identity is malformed.")
        if value.get("fingerprint_kind") != "sha256_first_middle_last_blocks_not_whole_file":
            raise ValueError("Saved dataset source fingerprint type is unsupported.")
        if not isinstance(value.get("partial_fingerprint"), str):
            raise TypeError("Saved dataset partial fingerprint must be text.")
        return value

    def append_analysis(self, analysis_name: str, parameters: dict[str, Any], summary: str):
        history = []
        if self.history_path.exists():
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        history.append({"analysis": analysis_name, "parameters": parameters,
                        "summary": summary,
                        "timestamp": datetime.now(timezone.utc).isoformat()})
        _atomic_json(self.history_path, history, prefix="analysis-history")
        self._touch_manifest()

    def load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def append_execution_record(self, record: ExecutionRecord) -> None:
        """Append a typed execution snapshot without replacing earlier results."""
        if not isinstance(record, ExecutionRecord):
            raise TypeError("record must be an ExecutionRecord.")
        history = self.load_execution_history()
        if any(item.get("execution_id") == record.execution_id for item in history):
            raise ValueError(f"Execution {record.execution_id} is already stored in this project.")
        history.append(record.to_snapshot())
        serialized = json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = self.root / f".execution-history-{uuid4()}.tmp"
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.execution_history_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._touch_manifest()

    def load_execution_history(self) -> list[dict[str, Any]]:
        """Read JSON-safe typed execution and approval snapshots."""
        if not self.execution_history_path.exists():
            return []
        value = json.loads(self.execution_history_path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("Project execution history must contain a list of record objects.")
        return value

    def save_pipeline(self, pipeline: object) -> Path:
        """Persist one immutable pipeline revision without replacing an earlier one."""
        from ..analysis.pipelines import PipelineDefinition

        if not isinstance(pipeline, PipelineDefinition):
            raise TypeError("pipeline must be a PipelineDefinition.")
        key = hashlib.sha256(pipeline.pipeline_id.encode("utf-8")).hexdigest()
        directory = self.pipelines_directory / key
        destination = directory / f"revision-{pipeline.revision:06d}.json"
        if destination.exists():
            raise FileExistsError(
                f"pipeline revision {pipeline.pipeline_id!r} r{pipeline.revision} already exists."
            )
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(destination, pipeline.model_dump(mode="json"), prefix="pipeline")
        self._touch_manifest()
        return destination

    def save_pipeline_revision(self, pipeline: object) -> Path:
        """Create and persist the next revision, never replacing an existing one."""
        from ..analysis.pipelines import PipelineDefinition, revise_pipeline

        if not isinstance(pipeline, PipelineDefinition):
            raise TypeError("pipeline must be a PipelineDefinition.")
        try:
            latest = self.load_pipeline(pipeline.pipeline_id)
        except FileNotFoundError:
            return self.save_pipeline(pipeline)
        candidate = revise_pipeline(
            latest,
            name=pipeline.name,
            description=pipeline.description,
            nodes=pipeline.nodes,
        )
        return self.save_pipeline(candidate)

    def load_pipeline(self, pipeline_id: str, revision: int | None = None):
        """Load and validate a saved pipeline; unsupported schemas are reported."""
        from ..analysis.pipelines import PipelineDefinition

        key = hashlib.sha256(pipeline_id.encode("utf-8")).hexdigest()
        directory = self.pipelines_directory / key
        if not directory.is_dir():
            raise FileNotFoundError(f"Saved pipeline {pipeline_id!r} was not found.")
        candidates = sorted(directory.glob("revision-*.json"))
        if revision is not None:
            candidates = [directory / f"revision-{revision:06d}.json"]
        if not candidates or not candidates[-1].is_file():
            raise FileNotFoundError(
                f"Saved pipeline revision {revision!r} for {pipeline_id!r} was not found."
            )
        return PipelineDefinition.model_validate_json(
            candidates[-1].read_text(encoding="utf-8"), strict=True
        )

    def list_pipelines(self) -> list[dict[str, Any]]:
        """Return latest validated revisions with explicit damaged-record errors."""
        results: list[dict[str, Any]] = []
        if not self.pipelines_directory.is_dir():
            return results
        for directory in sorted(self.pipelines_directory.iterdir()):
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("revision-*.json"))
            if not files:
                results.append({"status": "invalid", "path": str(directory), "reason": "no pipeline revisions found"})
                continue
            try:
                from ..analysis.pipelines import PipelineDefinition

                pipeline = PipelineDefinition.model_validate_json(
                    files[-1].read_text(encoding="utf-8"), strict=True
                )
                results.append({"status": "available", "pipeline": pipeline})
            except (OSError, ValueError, TypeError) as exc:
                results.append({"status": "invalid", "path": str(files[-1]), "reason": str(exc)})
        return results

    def save_pipeline_run(self, run_state: object) -> Path:
        """Atomically update a run's status snapshot without changing pipeline revisions."""
        from ..analysis.pipelines import PipelineRunState

        if not isinstance(run_state, PipelineRunState):
            raise TypeError("run_state must be a PipelineRunState.")
        key = hashlib.sha256(run_state.pipeline_id.encode("utf-8")).hexdigest()
        directory = self.pipelines_directory / key / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{run_state.run_id}.json"
        _atomic_json(destination, run_state.model_dump(mode="json"), prefix="pipeline-run")
        self._touch_manifest()
        return destination

    def list_pipeline_runs(self) -> list[dict[str, Any]]:
        """Validate saved pipeline-run snapshots without resuming execution."""
        from ..analysis.pipelines import PipelineRunState

        results: list[dict[str, Any]] = []
        if not self.pipelines_directory.is_dir():
            return results
        for directory in self.pipelines_directory.glob("*/runs/*.json"):
            try:
                state = PipelineRunState.model_validate_json(
                    directory.read_text(encoding="utf-8"), strict=True
                )
                results.append({"status": "available", "run": state})
            except (OSError, ValueError, TypeError) as exc:
                results.append({"status": "invalid", "path": str(directory), "reason": str(exc)})
        return results

    def source_match(self, profile: DatasetProfile) -> dict[str, object]:
        """Compare a candidate to the saved source; a partial match is never conclusive."""
        if not self.source_identity_path.is_file():
            return {"status": "identity_unavailable", "matches": False}
        try:
            saved = self.load_source_identity()
            if saved is None:
                return {"status": "identity_unavailable", "matches": False}
            path = Path(profile.source_path)
            stat = path.stat()
            candidate_fingerprint = _partial_file_fingerprint(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return {
                "status": "identity_unavailable", "matches": False,
                "reason": f"Saved source identity cannot be verified: {exc}",
            }
        structural_reasons = []
        if saved.get("file_format") != profile.file_format:
            structural_reasons.append("file format differs")
        if saved.get("dataset_path") != profile.dataset_path:
            structural_reasons.append("selected MATLAB dataset path differs")
        if saved.get("shape") != profile.shape:
            structural_reasons.append("shape differs")
        if saved.get("dtype") != profile.dtype:
            structural_reasons.append("data type differs")
        if saved.get("size_bytes") != stat.st_size:
            structural_reasons.append("file size differs")
        if saved.get("partial_fingerprint") != candidate_fingerprint:
            structural_reasons.append("bounded partial content fingerprint differs")
        structural_match = not structural_reasons
        return {
            "status": "partial_match_confirmation_required" if structural_match else "mismatch",
            "matches": structural_match,
            "fingerprint_kind": saved.get("fingerprint_kind"),
            "same_path": os.path.normcase(str(Path(saved.get("source_path", "")).resolve()))
            == os.path.normcase(str(path.resolve())),
            "mtime_changed": saved.get("mtime_ns") != stat.st_mtime_ns,
            "reasons": structural_reasons,
        }

    def record_source_resolution(self, status: str, details: dict[str, object]) -> None:
        """Persist uncertainty about an explicitly researcher-confirmed reconnection."""
        _atomic_json(
            self.source_resolution_path,
            {
                "status": status,
                "details": details,
                "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            prefix="source-resolution",
        )

    def discover_artifacts(self) -> list[dict[str, object]]:
        """Return artifact availability summaries without regenerating or deleting data."""
        entries: list[dict[str, object]] = []
        history = self.load_execution_history()
        executions = {str(item.get("execution_id")): item for item in history}
        for kind, store in (
            ("waveforms", self.artifacts),
            ("isi", self.isi_artifacts),
            ("mua", self.mua_artifacts),
        ):
            if not store.root.is_dir():
                continue
            for directory in sorted(store.root.iterdir()):
                if not directory.is_dir():
                    continue
                if directory.name.startswith("."):
                    entries.append({"kind": kind, "artifact_id": directory.name, "status": "incomplete_write"})
                    continue
                manifest_path = directory / "manifest.json"
                item: dict[str, object] = {"kind": kind, "artifact_id": directory.name}
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if not isinstance(manifest, dict):
                        raise TypeError("artifact manifest must contain a JSON object.")
                    schema_version = manifest.get("schema_version", "legacy-unspecified")
                    if schema_version not in ("legacy-unspecified", "1.0.0", 1):
                        item.update({"status": "unsupported_schema", "schema_version": schema_version})
                    else:
                        reference = f"artifact://{kind}/{directory.name}"
                        with store.open(reference):
                            pass
                        execution_id = manifest.get("execution_id")
                        execution = executions.get(str(execution_id)) if execution_id else None
                        if not execution_id:
                            item.update({
                                "status": "lineage_unverified",
                                "reason": "artifact manifest has no producing execution ID",
                            })
                        elif execution is None:
                            item.update({
                                "status": "lineage_unverified",
                                "reason": "producing execution is not present in project history",
                            })
                        elif execution is not None and (
                            execution.get("operation_id") != manifest.get("operation_id")
                            or execution.get("operation_version") != manifest.get("operation_version")
                            or execution.get("implementation_version") != manifest.get("implementation_version")
                            or _snapshot_artifact_reference(execution) != reference
                            or _snapshot_dataset_id(execution) != _manifest_dataset_id(manifest)
                        ):
                            item.update({
                                "status": "invalid_lineage",
                                "reason": "artifact reference or operation identity does not match its execution",
                            })
                        else:
                            item.update({"status": "available", "manifest": manifest})
                except (OSError, TypeError, ValueError, KeyError) as exc:
                    item.update({"status": "invalid", "reason": str(exc)})
                entries.append(item)
        return entries

    def restore_execution_records(self, registry) -> tuple[list[object], list[dict[str, str]]]:
        """Rehydrate compatible historical records; never recreate approvals or run code."""
        from ..analysis.operations import restore_execution_record

        records = []
        rejected = []
        for index, snapshot in enumerate(self.load_execution_history()):
            execution_id = str(snapshot.get("execution_id", f"record-{index}"))
            try:
                definition = registry.get(str(snapshot.get("operation_id", "")))
                records.append(restore_execution_record(snapshot, definition))
            except (OSError, TypeError, ValueError, KeyError) as exc:
                rejected.append({"execution_id": execution_id, "reason": str(exc)})
        return records, rejected


def _snapshot_artifact_reference(snapshot: dict[str, Any]) -> str | None:
    output = snapshot.get("output")
    if not isinstance(output, dict):
        return None
    for key in ("artifact_reference", "interval_artifact_reference"):
        value = output.get(key)
        if isinstance(value, str):
            return value
    return None


def _snapshot_dataset_id(snapshot: dict[str, Any]) -> str | None:
    provenance = snapshot.get("provenance")
    dataset = provenance.get("dataset_reference") if isinstance(provenance, dict) else None
    if isinstance(dataset, dict) and isinstance(dataset.get("dataset_id"), str):
        return dataset["dataset_id"]
    output = snapshot.get("output")
    if isinstance(output, dict) and isinstance(output.get("dataset_id"), str):
        return output["dataset_id"]
    return None


def _manifest_dataset_id(manifest: dict[str, Any]) -> str | None:
    dataset_id = manifest.get("dataset_id")
    if isinstance(dataset_id, str):
        return dataset_id
    dataset = manifest.get("dataset_reference")
    if isinstance(dataset, dict):
        value = dataset.get("dataset_id")
        return value if isinstance(value, str) else None
    return None
