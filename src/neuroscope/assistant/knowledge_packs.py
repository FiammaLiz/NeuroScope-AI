"""Portable, versioned scientific knowledge packs.

Archives are untrusted data. They contain no executable code, are validated
before installation, and are indexed through the existing local library
retriever. Pack/library scopes are organizational labels, not access control.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from ..analysis.contracts import OperationModel
from .library import (
    INDEX_VERSION,
    MAX_EXTRACTED_CHARS,
    MAX_SOURCE_BYTES,
    LibraryDocument,
    LibraryDocumentMetadata,
    LiteratureLibraryError,
    LiteratureLibraryStore,
    ScientificLibrary,
    _atomic_bytes,
    _atomic_json,
    _extract,
    _make_passages,
    _safe_filename,
    _utc,
)

KNOWLEDGE_PACK_SCHEMA_VERSION = 1
KNOWLEDGE_PACK_FORMAT = "neuroscope-knowledge-pack"
MAX_PACK_BYTES = 100 * 1024 * 1024
MAX_PACK_ENTRIES = 256
MAX_PACK_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PACK_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_SUPPLEMENT_CHARS = 100_000
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


class KnowledgePackError(ValueError):
    """A knowledge pack is invalid, unsafe, unavailable, or incompatible."""


class KnowledgePackDependency(OperationModel):
    pack_id: UUID
    version: str = Field(pattern=_SEMVER.pattern)
    optional: bool = True


class KnowledgePackResource(OperationModel):
    resource_id: UUID
    kind: Literal[
        "reference", "document", "methodology", "terminology", "protocol",
        "interpretation_warning", "researcher_note", "benchmark", "profile_reference",
    ]
    title: str = Field(min_length=1, max_length=500)
    metadata: LibraryDocumentMetadata = Field(default_factory=LibraryDocumentMetadata)
    source_library_id: UUID | None = None
    source_document_id: UUID | None = None
    source_provenance: tuple[str, ...] = Field(default=(), max_length=100)
    content_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    fingerprint_kind: Literal["source_bytes_sha256", "literature_record_json_sha256"] | None = None
    payload_path: str | None = None
    payload_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    payload_filename: str | None = Field(default=None, max_length=255)
    authored_content: str | None = Field(default=None, max_length=MAX_SUPPLEMENT_CHARS)
    associated_profile_id: str | None = Field(default=None, max_length=200)
    associated_profile_version: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_payload_contract(self):
        if bool(self.payload_path) != bool(self.payload_sha256):
            raise ValueError("payload path and payload fingerprint must be supplied together.")
        if self.payload_path is not None:
            path = PurePosixPath(self.payload_path)
            if path.is_absolute() or ".." in path.parts or "\\" in self.payload_path:
                raise ValueError("payload path must be a safe package-relative path.")
            if not self.payload_path.startswith("payloads/"):
                raise ValueError("payload path must be inside the package payloads directory.")
            if not self.payload_filename:
                raise ValueError("a payload filename is required when document bytes are included.")
        if self.authored_content is not None and self.kind not in {
            "methodology", "terminology", "protocol", "interpretation_warning",
            "researcher_note", "benchmark", "profile_reference",
        }:
            raise ValueError("authored content is only supported for methodological and note resources.")
        return self


class KnowledgePackManifest(OperationModel):
    format: Literal["neuroscope-knowledge-pack"] = KNOWLEDGE_PACK_FORMAT
    schema_version: int = KNOWLEDGE_PACK_SCHEMA_VERSION
    pack_id: UUID
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=4000)
    version: str = Field(pattern=_SEMVER.pattern)
    author: str | None = Field(default=None, max_length=300)
    organization: str | None = Field(default=None, max_length=300)
    scientific_domains: tuple[str, ...] = Field(default=(), max_length=50)
    created_at_utc: datetime
    modified_at_utc: datetime
    source_library_id: UUID | None = None
    resources: tuple[KnowledgePackResource, ...] = Field(max_length=500)
    associated_profile_ids: tuple[str, ...] = Field(default=(), max_length=100)
    dependencies: tuple[KnowledgePackDependency, ...] = Field(default=(), max_length=100)
    license_declaration: str | None = Field(default=None, max_length=500)
    redistribution_status: Literal["unknown", "researcher_confirmed", "not_permitted"] = "unknown"
    privacy_classification: Literal["public", "private", "unknown", "project_confidential"] = "unknown"
    minimum_neuroscope_version: str = "0.1.0"
    maximum_neuroscope_version: str | None = None
    indexing_version: str = INDEX_VERSION
    package_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self):
        if not _SEMVER.fullmatch(self.version):
            raise ValueError("pack version must use semantic version syntax.")
        if not _SEMVER.fullmatch(self.minimum_neuroscope_version):
            raise ValueError("minimum NeuroScope version must use semantic version syntax.")
        if self.maximum_neuroscope_version and not _SEMVER.fullmatch(self.maximum_neuroscope_version):
            raise ValueError("maximum NeuroScope version must use semantic version syntax.")
        if len({resource.resource_id for resource in self.resources}) != len(self.resources):
            raise ValueError("knowledge pack resource identifiers must be unique.")
        if len({(item.pack_id, item.version) for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("knowledge pack dependencies must be unique.")
        if any(item.pack_id == self.pack_id and item.version == self.version for item in self.dependencies):
            raise ValueError("a knowledge pack cannot depend on itself at the same version.")
        payload_paths = [resource.payload_path for resource in self.resources if resource.payload_path]
        if len(set(payload_paths)) != len(payload_paths):
            raise ValueError("knowledge pack payload paths must be unique.")
        return self


class KnowledgePackPreview(OperationModel):
    manifest: KnowledgePackManifest
    payload_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    installed: bool = False


class KnowledgePackService:
    """Create, inspect and install safe local knowledge-pack archives."""

    def __init__(self, library_store: LiteratureLibraryStore) -> None:
        self.library_store = library_store
        self.version_index_path = library_store.root / "knowledge_pack_versions.json"

    def _known_versions(self) -> dict:
        if not self.version_index_path.exists():
            return {"schema_version": 1, "versions": []}
        try:
            value = json.loads(self.version_index_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != 1 or not isinstance(value.get("versions"), list):
                raise ValueError
            return value
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            raise KnowledgePackError("knowledge pack version registry is damaged; refusing to change version records.") from exc

    def _record_version(self, manifest: KnowledgePackManifest) -> None:
        registry = self._known_versions()
        key = (str(manifest.pack_id), manifest.version)
        rows = registry["versions"]
        existing = [item for item in rows if (item.get("pack_id"), item.get("version")) == key]
        installed = [
            item for item in self.library_store.list_libraries()
            if item.knowledge_pack_id == manifest.pack_id
            and item.knowledge_pack_version == manifest.version
        ]
        if installed and any(item.knowledge_pack_content_sha256 != manifest.package_sha256 for item in installed):
            raise KnowledgePackError("an installed Knowledge Pack version cannot be changed or replaced with different contents.")
        if existing and any(item.get("package_sha256") != manifest.package_sha256 for item in existing):
            raise KnowledgePackError("a knowledge pack version is immutable and this ID/version was already recorded with different contents.")
        if not existing:
            rows.append({
                "pack_id": str(manifest.pack_id),
                "version": manifest.version,
                "package_sha256": manifest.package_sha256,
                "recorded_at_utc": _utc().isoformat(),
            })
            _atomic_json(self.version_index_path, registry)

    def create_from_library(
        self,
        library_id: UUID | str,
        document_ids: tuple[UUID | str, ...],
        destination: str | Path,
        *,
        name: str,
        version: str,
        pack_id: UUID,
        description: str = "",
        author: str | None = None,
        organization: str | None = None,
        scientific_domains: tuple[str, ...] = (),
        supplements: tuple[KnowledgePackResource, ...] = (),
        associated_profile_ids: tuple[str, ...] = (),
        dependencies: tuple[KnowledgePackDependency, ...] = (),
        privacy_classification: Literal["public", "private", "unknown", "project_confidential"] | None = None,
        license_declaration: str | None = None,
        minimum_neuroscope_version: str = "0.1.0",
        maximum_neuroscope_version: str | None = None,
        include_document_payloads: bool = False,
        researcher_confirmed_redistribution: bool = False,
        researcher_confirmed_private_payloads: bool = False,
    ) -> Path:
        """Create a pack from an exact resource selection; metadata-only by default."""
        if not document_ids and not supplements:
            raise KnowledgePackError("select at least one reference or authored resource.")
        if len({str(item) for item in document_ids}) != len(document_ids):
            raise KnowledgePackError("selected document identifiers must be unique.")
        library = self.library_store.get_library(library_id)
        selected = {UUID(str(item)) for item in document_ids}
        if not selected.issubset(set(library.document_ids)):
            raise KnowledgePackError("a selected document is not in the chosen library.")
        selected_documents = [
            self.library_store.get_document(library.library_id, item)
            for item in document_ids
        ]
        access_states = {item.metadata.access_status for item in selected_documents}
        privacy_states = access_states | {item.metadata.access_status for item in supplements}
        if privacy_classification == "public" and (access_states - {"public"} or any(item.metadata.access_status != "public" for item in supplements)):
            raise KnowledgePackError("a pack with private/unknown documents or researcher notes cannot be labeled public without resolving their privacy status.")
        if privacy_classification is None:
            privacy_classification = (
                "private" if "private" in privacy_states
                else "public" if access_states == {"public"} and not supplements
                else "unknown"
            )
        elif privacy_classification == "unknown" and "private" in privacy_states:
            privacy_classification = "private"
        resources: list[KnowledgePackResource] = []
        payloads: dict[str, bytes] = {}
        private_payload_present = False
        for document_id in document_ids:
            document = self.library_store.get_document(library.library_id, document_id)
            resource_id = document.document_id
            payload_path = payload_hash = payload_filename = None
            fingerprint = document.content_sha256
            if include_document_payloads:
                if not researcher_confirmed_redistribution:
                    raise KnowledgePackError("content-inclusive export requires explicit confirmation that document redistribution is authorized.")
                if document.metadata.access_status != "public":
                    private_payload_present = True
                if not document.source_retained or not document.source_relative_path:
                    raise KnowledgePackError(f"resource {document.document_id} has no retained source payload; export it reference-only.")
                source = self.library_store._safe_library_path(library, document.source_relative_path)
                if not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
                    raise KnowledgePackError(f"resource {document.document_id} is missing or exceeds the document payload limit.")
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != document.content_sha256:
                    raise KnowledgePackError(f"resource {document.document_id} source fingerprint changed; review and explicitly replace it first.")
                payload_filename = _safe_filename(document.source_filename)
                payload_path = f"payloads/{resource_id}/{payload_filename}"
                payload_hash = hashlib.sha256(payload).hexdigest()
                payloads[payload_path] = payload
            resources.append(KnowledgePackResource(
                resource_id=resource_id,
                kind="reference" if document.extraction_status == "metadata_only" else "document",
                title=document.display_title,
                metadata=_portable_metadata(document.metadata).model_copy(update={
                    "evidence_level": "full_text" if payload_path else "metadata_only"
                }),
                source_library_id=library.library_id,
                source_document_id=document.document_id,
                source_provenance=tuple(item for value in document.provenance if (item := _portable_provenance(value)) is not None),
                content_fingerprint=fingerprint,
                fingerprint_kind=document.fingerprint_kind,
                payload_path=payload_path,
                payload_sha256=payload_hash,
                payload_filename=payload_filename,
            ))
        if private_payload_present and not researcher_confirmed_private_payloads:
            raise KnowledgePackError("private or access-unknown content requires a separate explicit confirmation.")
        for resource in supplements:
            if resource.kind in {"reference", "document"}:
                raise KnowledgePackError("supplements must be notes, methods, protocols, warnings, benchmarks, terminology, or profile references.")
            if include_document_payloads and resource.authored_content is not None and resource.metadata.access_status != "public":
                private_payload_present = True
            resources.append(resource)
        if private_payload_present and include_document_payloads and not researcher_confirmed_private_payloads:
            raise KnowledgePackError("private or access-unknown content requires a separate explicit confirmation.")
        now = datetime.now(timezone.utc)
        empty = KnowledgePackManifest(
            pack_id=pack_id,
            name=name.strip(),
            description=description,
            version=version,
            author=author,
            organization=organization,
            scientific_domains=scientific_domains,
            created_at_utc=now,
            modified_at_utc=now,
            source_library_id=library.library_id,
            resources=tuple(resources),
            associated_profile_ids=associated_profile_ids,
            dependencies=dependencies,
            license_declaration=license_declaration,
            redistribution_status="researcher_confirmed" if include_document_payloads and researcher_confirmed_redistribution else "unknown",
            privacy_classification=privacy_classification,
            minimum_neuroscope_version=minimum_neuroscope_version,
            maximum_neuroscope_version=maximum_neuroscope_version,
            package_sha256="0" * 64,
        )
        package_digest = _package_digest(empty, payloads)
        manifest = empty.model_copy(update={"package_sha256": package_digest})
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("knowledge pack destination already exists; refusing to overwrite.")
        if len(payloads) + 1 > MAX_PACK_ENTRIES:
            raise KnowledgePackError("knowledge pack exceeds the archive entry-count limit; reduce selected payloads.")
        if len(_manifest_bytes(manifest)) > MAX_PACK_MANIFEST_BYTES:
            raise KnowledgePackError("knowledge pack manifest exceeds its size limit; reduce authored text or selected metadata.")
        if sum(len(data) for data in payloads.values()) + len(manifest.model_dump_json().encode()) > MAX_PACK_BYTES:
            raise KnowledgePackError("knowledge pack exceeds the configured size limit.")
        self._record_version(manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", _manifest_bytes(manifest))
                for path, data in sorted(payloads.items()):
                    archive.writestr(path, data)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    def inspect(self, source: str | Path) -> KnowledgePackPreview:
        manifest, payloads = _read_package(source)
        warnings = []
        if not _compatible_with_application(manifest):
            warnings.append("This pack declares an incompatible NeuroScope application version.")
        for dependency in manifest.dependencies:
            if not dependency.optional and not any(
                item.knowledge_pack_id == dependency.pack_id and item.knowledge_pack_version == dependency.version
                for item in self.library_store.list_libraries()
            ):
                warnings.append(f"Required pack dependency {dependency.pack_id} {dependency.version} is not installed.")
        if manifest.redistribution_status == "unknown" and payloads:
            warnings.append("Redistribution permission is unknown; inspect rights before using or sharing payloads.")
        if manifest.privacy_classification != "public" and payloads:
            warnings.append("This pack contains non-public or access-status-unknown content.")
        if manifest.license_declaration is None:
            warnings.append("Pack license is unknown; no redistribution permission is implied.")
        if any(resource.metadata.publication_status == "preprint" for resource in manifest.resources):
            warnings.append("Preprint — not peer reviewed.")
        return KnowledgePackPreview(manifest=manifest, payload_count=len(payloads), warnings=tuple(warnings))

    def import_pack(
        self,
        source: str | Path,
        *,
        activate: bool = False,
        scope: Literal["core", "personal", "laboratory", "project"] = "project",
    ) -> ScientificLibrary:
        """Validate the whole package, then install it as a separate local library."""
        if scope not in {"core", "personal", "laboratory", "project"}:
            raise KnowledgePackError("installed library scope must be core, personal, laboratory, or project.")
        manifest, payloads = _read_package(source)
        if not _compatible_with_application(manifest):
            raise KnowledgePackError("knowledge pack is not compatible with this NeuroScope application version.")
        installed_pairs = {
            (item.knowledge_pack_id, item.knowledge_pack_version)
            for item in self.library_store.list_libraries()
            if item.knowledge_pack_id is not None
        }
        missing_required = [
            item for item in manifest.dependencies
            if not item.optional and (item.pack_id, item.version) not in installed_pairs
        ]
        if missing_required:
            raise KnowledgePackError("a required Knowledge Pack dependency is not installed; pack was not imported.")
        self._record_version(manifest)
        existing = [
            library for library in self.library_store.list_libraries()
            if library.knowledge_pack_id == manifest.pack_id and library.knowledge_pack_version == manifest.version
        ]
        if existing:
            prior = existing[0]
            if prior.knowledge_pack_content_sha256 != manifest.package_sha256:
                raise KnowledgePackError("this pack ID/version already exists with different contents; versions are immutable.")
            raise KnowledgePackError("this knowledge pack version is already installed; refusing to overwrite it.")
        library_id = uuid5(manifest.pack_id, f"{manifest.version}:{manifest.package_sha256}")
        library = ScientificLibrary(
            library_id=library_id,
            name=f"{manifest.name} ({manifest.version})",
            scope=scope,
            created_at_utc=_utc(),
            document_ids=(),
            provenance=(f"knowledge_pack:{manifest.pack_id}", f"version:{manifest.version}", f"sha256:{manifest.package_sha256}"),
            knowledge_pack_id=manifest.pack_id,
            knowledge_pack_version=manifest.version,
            knowledge_pack_content_sha256=manifest.package_sha256,
        )
        prepared = []
        for resource in manifest.resources:
            doc_id = uuid5(manifest.pack_id, f"{manifest.version}:{resource.resource_id}")
            content = None
            suffix = ".md"
            if resource.payload_path:
                content = payloads[resource.payload_path]
                suffix = Path(resource.payload_filename or "resource.txt").suffix.lower()
            elif resource.authored_content is not None:
                content = resource.authored_content.encode("utf-8")
            passages = []
            status = "manifest_only"
            warnings = []
            if content is not None:
                if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
                    raise KnowledgePackError(f"resource {resource.title!r} has an unsupported payload format.")
                extracted, warnings = _extract(suffix, content)
                remaining = MAX_EXTRACTED_CHARS
                bounded = []
                for page, text in extracted:
                    bounded_text = text[:remaining]
                    remaining -= len(bounded_text)
                    if len(bounded_text) < len(text):
                        warnings.append("pack extraction truncated at the configured document character bound.")
                    if bounded_text:
                        bounded.append((page, bounded_text))
                    if remaining <= 0:
                        break
                passages = _make_passages(doc_id, bounded)
                if not passages:
                    raise KnowledgePackError(f"resource {resource.title!r} has no extractable text.")
                status = "indexed"
            metadata = resource.metadata
            if resource.authored_content is not None:
                metadata = metadata.model_copy(update={
                    "source_type": "manual",
                    "evidence_level": "full_text",
                    "publication_status": "unknown",
                })
            elif content is None:
                metadata = metadata.model_copy(update={"evidence_level": "metadata_only"})
            document = LibraryDocument(
                document_id=doc_id,
                library_id=library_id,
                display_title=resource.title,
                source_filename=resource.payload_filename or f"{resource.resource_id}.metadata.json",
                metadata=metadata,
                imported_at_utc=_utc(),
                content_sha256=(
                    resource.payload_sha256
                    or (hashlib.sha256(resource.authored_content.encode("utf-8")).hexdigest() if resource.authored_content is not None else None)
                    or resource.content_fingerprint
                    or hashlib.sha256(b"").hexdigest()
                ),
                fingerprint_kind=resource.fingerprint_kind or "source_bytes_sha256",
                extraction_status=status,
                extraction_warnings=tuple(warnings),
                assistant_visible=False,
                source_retained=content is not None,
                source_relative_path=f"sources/{doc_id}/{_safe_filename(resource.payload_filename or (str(resource.resource_id) + suffix))}" if content is not None else None,
                source_path_private=None,
                index_version=INDEX_VERSION,
                provenance=(
                    "imported_from_knowledge_pack",
                    f"pack_id:{manifest.pack_id}",
                    f"pack_version:{manifest.version}",
                    f"pack_sha256:{manifest.package_sha256}",
                    f"pack_resource_id:{resource.resource_id}",
                    *(f"source:{item}" for item in resource.source_provenance),
                ),
            )
            prepared.append((document, tuple(passages), content))
        # All archive entries, hashes, schemas, and text extraction have passed
        # before the target catalog is changed.
        self.library_store.root.mkdir(parents=True, exist_ok=True)
        self.library_store._empty = False
        self.library_store._library_dir(library_id).mkdir(parents=True, exist_ok=False)
        try:
            manifest_path = self.library_store._library_dir(library_id) / "knowledge_pack.json"
            _atomic_json(
                manifest_path,
                manifest.model_dump(mode="json"),
            )
            manifest_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for document, passages, content in prepared:
                document_dir = self.library_store._document_dir(library_id, document.document_id)
                document_dir.mkdir(parents=True, exist_ok=False)
                _atomic_json(document_dir / "record.json", document.model_dump(mode="json"))
                _atomic_json(document_dir / "passages.json", [item.model_dump(mode="json") for item in passages])
                if content is not None:
                    source_path = self.library_store._safe_library_path(library, document.source_relative_path)
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_bytes(source_path, content)
            installed = library.model_copy(update={
                "document_ids": tuple(item[0].document_id for item in prepared),
                "provenance": (*library.provenance, f"manifest_sha256:{manifest_fingerprint}"),
            })
            catalog = self.library_store._load_catalog()
            catalog["libraries"].append(installed.model_dump(mode="json"))
            _atomic_json(self.library_store.catalog_path, catalog)
        except Exception:
            import shutil
            shutil.rmtree(self.library_store._library_dir(library_id), ignore_errors=True)
            raise
        if activate:
            active = tuple(self.library_store.active_library_ids()) + (str(library_id),)
            self.library_store.set_active_libraries(active)
        return self.library_store.get_library(library_id)

    def installed_manifest(
        self, pack_id: UUID | str, version: str, *, verify_payloads: bool = True
    ) -> KnowledgePackManifest:
        library = self._find_pack(pack_id, version)
        path = self.library_store._library_dir(library.library_id) / "knowledge_pack.json"
        try:
            manifest_bytes = path.read_bytes()
            manifest = KnowledgePackManifest.model_validate_json(manifest_bytes, strict=True)
        except (OSError, ValueError) as exc:
            raise KnowledgePackError("installed Knowledge Pack metadata is damaged or unavailable.") from exc
        expected_manifest_sha = next((item.partition(":")[2] for item in library.provenance if item.startswith("manifest_sha256:")), None)
        if expected_manifest_sha is None or hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
            raise KnowledgePackError("installed Knowledge Pack manifest fingerprint does not match its registered identity.")
        if manifest.pack_id != library.knowledge_pack_id or manifest.package_sha256 != library.knowledge_pack_content_sha256:
            raise KnowledgePackError("installed Knowledge Pack metadata does not match its registered identity.")
        if not verify_payloads:
            return manifest
        payloads: dict[str, bytes] = {}
        try:
            for resource in manifest.resources:
                if not resource.payload_path:
                    continue
                document_id = uuid5(manifest.pack_id, f"{manifest.version}:{resource.resource_id}")
                document = self.library_store.get_document(library.library_id, document_id)
                source = self.library_store._safe_library_path(library, document.source_relative_path)
                content = source.read_bytes()
                if hashlib.sha256(content).hexdigest() != resource.payload_sha256:
                    raise KnowledgePackError(f"installed payload fingerprint mismatch for resource {resource.resource_id}.")
                payloads[resource.payload_path] = content
        except (OSError, LiteratureLibraryError) as exc:
            raise KnowledgePackError("installed Knowledge Pack payload is missing or cannot be verified.") from exc
        if _package_digest(manifest, payloads) != manifest.package_sha256:
            raise KnowledgePackError("installed Knowledge Pack contents do not match their recorded fingerprint.")
        return manifest

    def set_active(self, pack_id: UUID | str, version: str, active: bool) -> ScientificLibrary:
        library = self._find_pack(pack_id, version)
        selected = list(self.library_store.active_library_ids())
        if active:
            if str(library.library_id) not in selected:
                selected.append(str(library.library_id))
        else:
            selected = [item for item in selected if item != str(library.library_id)]
        self.library_store.set_active_libraries(tuple(selected))
        return library

    def remove(self, pack_id: UUID | str, version: str) -> None:
        library = self._find_pack(pack_id, version)
        self.library_store.remove_library(library.library_id)

    def _find_pack(self, pack_id: UUID | str, version: str) -> ScientificLibrary:
        matches = [
            item for item in self.library_store.list_libraries()
            if item.knowledge_pack_id == UUID(str(pack_id)) and item.knowledge_pack_version == version
        ]
        if len(matches) != 1:
            raise KnowledgePackError("selected knowledge pack version is not installed or is ambiguous.")
        return matches[0]


def _manifest_bytes(manifest: KnowledgePackManifest) -> bytes:
    return json.dumps(manifest.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _portable_provenance(value: str) -> str | None:
    if re.search(r"(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\|/(?:Users|home)/)", value):
        return None
    if value.lower().startswith(("file:", "local_path:")):
        return None
    return value[:1000]


def _portable_metadata(metadata: LibraryDocumentMetadata) -> LibraryDocumentMetadata:
    url = metadata.source_record_url
    if url and not url.lower().startswith(("http://", "https://")):
        url = None
    return metadata.model_copy(update={
        "source_record_url": url,
        "metadata_provenance": tuple(
            item for value in metadata.metadata_provenance
            if (item := _portable_provenance(value)) is not None
        ),
    })


def _package_digest(manifest: KnowledgePackManifest, payloads: dict[str, bytes]) -> str:
    data = manifest.model_dump(mode="json")
    data["package_sha256"] = ""
    # Creation/modification timestamps describe packaging activity, not the
    # immutable scientific resource set for a semantic pack version.
    data["created_at_utc"] = ""
    data["modified_at_utc"] = ""
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for path, content in sorted(payloads.items()):
        digest.update(path.encode("utf-8"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _compatible_with_application(manifest: KnowledgePackManifest) -> bool:
    try:
        current = importlib.metadata.version("neuroscope-ai")
    except importlib.metadata.PackageNotFoundError:
        current = "0.1.0"
    def key(version: str) -> tuple[int, int, int]:
        values = version.split("+", 1)[0].split("-", 1)[0].split(".")
        return tuple(int(item) for item in values[:3])
    current_key = key(current)
    return key(manifest.minimum_neuroscope_version) <= current_key and (
        manifest.maximum_neuroscope_version is None or current_key <= key(manifest.maximum_neuroscope_version)
    )


def _read_package(source: str | Path) -> tuple[KnowledgePackManifest, dict[str, bytes]]:
    path = Path(source)
    try:
        if not path.is_file() or path.stat().st_size > MAX_PACK_BYTES:
            raise KnowledgePackError("knowledge pack is missing or exceeds the package size limit.")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_PACK_ENTRIES:
                raise KnowledgePackError("knowledge pack has no files or exceeds the entry-count limit.")
            names = [item.filename for item in entries]
            if len(set(names)) != len(names):
                raise KnowledgePackError("knowledge pack contains duplicate archive paths.")
            if "manifest.json" not in names:
                raise KnowledgePackError("knowledge pack does not contain manifest.json.")
            total = 0
            for item in entries:
                name = item.filename
                path_parts = PurePosixPath(name)
                mode = item.external_attr >> 16
                if (path_parts.is_absolute() or ".." in path_parts.parts or "\\" in name
                        or name.startswith("/") or not name or stat.S_ISLNK(mode)):
                    raise KnowledgePackError("knowledge pack contains an unsafe archive path or link.")
                if item.flag_bits & 0x1:
                    raise KnowledgePackError("encrypted knowledge pack entries are unsupported.")
                total += item.file_size
                if total > MAX_PACK_UNCOMPRESSED_BYTES:
                    raise KnowledgePackError("knowledge pack exceeds the uncompressed size limit.")
                if item.compress_size == 0 and item.file_size > 0:
                    raise KnowledgePackError("knowledge pack has an invalid compression ratio.")
                if item.compress_size and item.file_size / item.compress_size > MAX_COMPRESSION_RATIO:
                    raise KnowledgePackError("knowledge pack exceeds the compression-ratio limit.")
            info = archive.getinfo("manifest.json")
            if info.file_size > MAX_PACK_MANIFEST_BYTES:
                raise KnowledgePackError("knowledge pack manifest exceeds its size limit.")
            raw = json.loads(archive.read(info).decode("utf-8"))
            if not isinstance(raw, dict):
                raise KnowledgePackError("knowledge pack manifest must be a JSON object.")
            if raw.get("format") != KNOWLEDGE_PACK_FORMAT:
                raise KnowledgePackError("file is not a NeuroScope Knowledge Pack.")
            if raw.get("schema_version") != KNOWLEDGE_PACK_SCHEMA_VERSION:
                raise KnowledgePackError(f"knowledge pack schema {raw.get('schema_version')!r} is unsupported.")
            manifest = KnowledgePackManifest.model_validate_json(json.dumps(raw), strict=True)
            if manifest.indexing_version != INDEX_VERSION:
                raise KnowledgePackError(f"knowledge pack indexing version {manifest.indexing_version!r} is unsupported.")
            if info.file_size > MAX_PACK_MANIFEST_BYTES:
                raise KnowledgePackError("knowledge pack manifest exceeds its size limit.")
            payloads: dict[str, bytes] = {}
            expected_names = {"manifest.json"}
            for resource in manifest.resources:
                if resource.payload_path:
                    expected = f"payloads/{resource.resource_id}/{_safe_filename(resource.payload_filename or '')}"
                    if resource.payload_path != expected:
                        raise KnowledgePackError("payload path does not match its resource identity and safe filename.")
                    expected_names.add(expected)
                    payload = archive.read(expected)
                    if len(payload) > MAX_SOURCE_BYTES:
                        raise KnowledgePackError("a document payload exceeds the per-document size limit.")
                    if hashlib.sha256(payload).hexdigest() != resource.payload_sha256:
                        raise KnowledgePackError(f"payload fingerprint mismatch for resource {resource.resource_id}.")
                    payloads[expected] = payload
            if set(names) != expected_names:
                raise KnowledgePackError("knowledge pack contains unreferenced or missing archive entries.")
            if _package_digest(manifest, payloads) != manifest.package_sha256:
                raise KnowledgePackError("knowledge pack manifest/content fingerprint does not match its contents.")
            return manifest, payloads
    except KnowledgePackError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise KnowledgePackError("knowledge pack is malformed, unreadable, or fails schema validation.") from exc
