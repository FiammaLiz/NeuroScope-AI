"""Persistent local scientific document libraries and deterministic retrieval.

Document text is treated as untrusted data. Library scopes are organizational
labels only; this module does not implement user authentication or encryption.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from ..analysis.contracts import OperationModel

LIBRARY_SCHEMA_VERSION = 1
INDEX_VERSION = "lexical-bm25-v1"
MAX_SOURCE_BYTES = 40 * 1024 * 1024
MAX_EXTRACTED_CHARS = 2_000_000
MAX_PASSAGE_CHARS = 1400
MAX_QUERY_CHARS = 500
MAX_RESULTS = 20
MAX_SEARCH_PASSAGES = 50_000
MAX_SEARCH_SECONDS = 3.0
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


class LiteratureLibraryError(ValueError):
    """Library data is invalid, unavailable, or cannot safely be used."""


class LibraryScope(str):
    CORE = "core"
    PERSONAL = "personal"
    LABORATORY = "laboratory"
    PROJECT = "project"


class LibraryDocumentMetadata(OperationModel):
    title: str | None = Field(default=None, max_length=500)
    doi: str | None = Field(default=None, max_length=200)
    authors: tuple[str, ...] = Field(default=(), max_length=100)
    publication_date: str | None = Field(default=None, max_length=40)
    source_type: Literal["journal_article", "preprint", "book", "manual", "notes", "other", "unknown"] = "unknown"
    publication_status: Literal["peer_reviewed", "preprint", "unverified", "unknown"] = "unknown"
    access_status: Literal["public", "private", "unknown"] = "unknown"
    license: str | None = Field(default=None, max_length=300)
    evidence_level: Literal["full_text", "abstract_only", "metadata_only"] = "full_text"
    linked_journal_doi: str | None = Field(default=None, max_length=200)
    identifier_verification: Literal["crossref_verified", "source_asserted", "unverified", "not_applicable"] = "not_applicable"
    identifier_source: str | None = Field(default=None, max_length=100)
    source_record_url: str | None = Field(default=None, max_length=1000)
    source_retrieved_at_utc: datetime | None = None
    publication_status_source: str | None = Field(default=None, max_length=200)
    metadata_provenance: tuple[str, ...] = Field(default=(), max_length=30)


class LibraryPassage(OperationModel):
    passage_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: UUID
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=200)
    ordinal: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_stop: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=MAX_PASSAGE_CHARS)

    @model_validator(mode="after")
    def valid_range(self):
        if self.char_stop <= self.char_start:
            raise ValueError("passage character range must be nonempty and half-open.")
        return self


class LibraryDocument(OperationModel):
    document_id: UUID
    library_id: UUID
    display_title: str = Field(min_length=1, max_length=500)
    source_filename: str = Field(min_length=1, max_length=255)
    metadata: LibraryDocumentMetadata
    imported_at_utc: datetime
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fingerprint_kind: Literal["source_bytes_sha256", "literature_record_json_sha256"] = "source_bytes_sha256"
    document_schema_version: int = Field(default=1, ge=1)
    document_revision: int = Field(default=1, ge=1)
    extraction_status: Literal["indexed", "metadata_only", "manifest_only", "changed_needs_review", "source_missing", "failed"]
    extraction_warnings: tuple[str, ...] = Field(default=(), max_length=30)
    assistant_visible: bool = False
    source_retained: bool = False
    source_relative_path: str | None = None
    source_path_private: str | None = None
    index_version: str = INDEX_VERSION
    provenance: tuple[str, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def location_is_relative(self):
        if self.source_relative_path is not None:
            path = PurePosixPath(self.source_relative_path)
            if path.is_absolute() or ".." in path.parts or "\\" in self.source_relative_path:
                raise ValueError("document storage paths must be safe library-relative paths.")
        return self


class ScientificLibrary(OperationModel):
    library_id: UUID
    name: str = Field(min_length=1, max_length=120)
    scope: Literal["core", "personal", "laboratory", "project"]
    created_at_utc: datetime
    indexing_version: str = INDEX_VERSION
    document_ids: tuple[UUID, ...] = ()
    provenance: tuple[str, ...] = ()
    knowledge_pack_id: UUID | None = None
    knowledge_pack_version: str | None = None
    knowledge_pack_content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def valid_pack_identity(self):
        if self.knowledge_pack_id is None:
            if self.knowledge_pack_version is not None or self.knowledge_pack_content_sha256 is not None:
                raise ValueError("Knowledge Pack version and fingerprint require a pack identifier.")
        elif self.knowledge_pack_version is None or self.knowledge_pack_content_sha256 is None:
            raise ValueError("Knowledge Pack identity requires version and content fingerprint.")
        return self


class RetrievedLibraryPassage(OperationModel):
    library_id: UUID
    library_name: str
    scope: Literal["core", "personal", "laboratory", "project"]
    document: LibraryDocument
    passage: LibraryPassage
    score: float = Field(ge=0)
    retriever_id: str = "neuroscope.local-bm25"
    retriever_version: str = INDEX_VERSION
    retrieved_at_utc: datetime
    provenance: tuple[str, ...]
    knowledge_pack_id: UUID | None = None
    knowledge_pack_version: str | None = None


class LibraryManifest(OperationModel):
    manifest_schema_version: int = LIBRARY_SCHEMA_VERSION
    library: ScientificLibrary
    documents: tuple[LibraryDocument, ...]
    exported_at_utc: datetime
    contains_document_payloads: Literal[False] = False

    @model_validator(mode="after")
    def consistent_members(self):
        if any(item.library_id != self.library.library_id for item in self.documents):
            raise ValueError("manifest document refers to another library.")
        if len({item.document_id for item in self.documents}) != len(self.documents):
            raise ValueError("manifest document identifiers must be unique.")
        return self


class LiteratureLibraryStore:
    """Atomic, local catalog. One root may contain multiple isolated libraries."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        self.root = Path(root).resolve()
        self.catalog_path = self.root / "catalog.json"
        self.selection_path = self.root / "active_corpus.json"
        self._create = create
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            self._empty = True
            return
        self._empty = False
        if not self.catalog_path.exists() and create:
            _atomic_json(self.catalog_path, {"schema_version": 1, "libraries": []})
        self._load_catalog()

    def _load_catalog(self) -> dict:
        if self._empty or not self.catalog_path.exists():
            return {"schema_version": 1, "libraries": []}
        try:
            value = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LiteratureLibraryError("library catalog is unreadable or malformed.") from exc
        if not isinstance(value, dict) or value.get("schema_version") != LIBRARY_SCHEMA_VERSION or not isinstance(value.get("libraries"), list):
            raise LiteratureLibraryError("library catalog schema is unsupported or malformed.")
        seen = set()
        for entry in value["libraries"]:
            try:
                lib = ScientificLibrary.model_validate_json(json.dumps(entry), strict=True)
            except ValueError as exc:
                raise LiteratureLibraryError("library catalog contains an invalid library record.") from exc
            if str(lib.library_id) in seen:
                raise LiteratureLibraryError("library catalog contains duplicate identifiers.")
            seen.add(str(lib.library_id))
        return value

    def list_libraries(self) -> tuple[ScientificLibrary, ...]:
        return tuple(ScientificLibrary.model_validate_json(json.dumps(item), strict=True) for item in self._load_catalog()["libraries"])

    def active_library_ids(self) -> tuple[str, ...]:
        if not self.selection_path.is_file():
            return ()
        try:
            value = json.loads(self.selection_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != 1 or not isinstance(value.get("library_ids"), list):
                raise ValueError
            available = {str(item.library_id) for item in self.list_libraries()}
            selected = tuple(str(UUID(item)) for item in value["library_ids"])
            if len(set(selected)) != len(selected) or any(item not in available for item in selected):
                raise ValueError
            return selected
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise LiteratureLibraryError("active library selection is damaged or references unavailable libraries.") from exc

    def set_active_libraries(self, library_ids: tuple[UUID | str, ...]) -> None:
        selected = tuple(dict.fromkeys(str(self.get_library(item).library_id) for item in library_ids))
        _atomic_json(self.selection_path, {"schema_version": 1, "library_ids": list(selected)})

    def create_library(self, name: str, scope: str) -> ScientificLibrary:
        if scope not in {"core", "personal", "laboratory", "project"}:
            raise LiteratureLibraryError("scope must be core, personal, laboratory, or project.")
        library = ScientificLibrary(library_id=uuid4(), name=name.strip(), scope=scope, created_at_utc=_utc())
        self.root.mkdir(parents=True, exist_ok=True)
        self._empty = False
        catalog = self._load_catalog()
        catalog["libraries"].append(library.model_dump(mode="json"))
        _atomic_json(self.catalog_path, catalog)
        self._library_dir(library.library_id).mkdir(parents=True, exist_ok=False)
        return library

    def get_library(self, library_id: UUID | str) -> ScientificLibrary:
        key = str(library_id)
        for item in self.list_libraries():
            if str(item.library_id) == key:
                return item
        raise LiteratureLibraryError("selected literature library is unavailable.")

    def remove_library(self, library_id: UUID | str) -> None:
        """Remove exactly one selected library and its local contents."""
        library = self.get_library(library_id)
        selected = tuple(item for item in self.active_library_ids() if item != str(library.library_id))
        library_dir = self._library_dir(library.library_id)
        if library_dir.exists():
            shutil.rmtree(library_dir)
        catalog = self._load_catalog()
        catalog["libraries"] = [
            item for item in catalog["libraries"]
            if item.get("library_id") != str(library.library_id)
        ]
        _atomic_json(self.catalog_path, catalog)
        self.set_active_libraries(tuple(UUID(item) for item in selected))

    def list_documents(self, library_id: UUID | str) -> tuple[LibraryDocument, ...]:
        library = self.get_library(library_id)
        rows = []
        for document_id in library.document_ids:
            path = self._document_path(library.library_id, document_id)
            try:
                rows.append(LibraryDocument.model_validate_json(path.read_text(encoding="utf-8"), strict=True))
            except (OSError, ValueError) as exc:
                raise LiteratureLibraryError(f"document {document_id} metadata or index is unavailable.") from exc
        return tuple(rows)

    def validate_existing(self) -> tuple[dict[str, str], ...]:
        """Inspect persisted records and indexes without extracting or rebuilding them."""
        statuses = []
        for library in self.list_libraries():
            for document in self.list_documents(library.library_id):
                try:
                    current = self.get_document(library.library_id, document.document_id)
                    if current.extraction_status == "changed_needs_review":
                        statuses.append({"document_id": str(current.document_id), "status": "changed_needs_review"})
                    elif current.extraction_status == "indexed":
                        self.load_passages(library.library_id, current.document_id)
                        statuses.append({"document_id": str(current.document_id), "status": "available"})
                    else:
                        statuses.append({"document_id": str(current.document_id), "status": current.extraction_status})
                except LiteratureLibraryError:
                    statuses.append({"document_id": str(document.document_id), "status": "corrupt_or_unavailable"})
        return tuple(statuses)

    def import_document(
        self,
        library_id: UUID | str,
        source_path: str | Path,
        *,
        metadata: LibraryDocumentMetadata | None = None,
        assistant_visible: bool = False,
        retain_original: bool = False,
        document_id: UUID | None = None,
        replace_existing: bool = False,
    ) -> LibraryDocument:
        library = self.get_library(library_id)
        if library.knowledge_pack_id is not None:
            raise LiteratureLibraryError("Knowledge Pack contents are immutable; import or create a new pack version instead of replacing documents.")
        source = Path(source_path)
        if not source.is_file():
            raise LiteratureLibraryError("selected source document does not exist or is not a file.")
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise LiteratureLibraryError(f"document exceeds the {MAX_SOURCE_BYTES}-byte ingestion limit.")
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        existing = next((d for d in self.list_documents(library.library_id) if document_id and d.document_id == document_id), None)
        if existing and not replace_existing:
            raise LiteratureLibraryError("document already exists; explicit replace_existing=True is required.")
        if replace_existing and existing is None:
            raise LiteratureLibraryError("replacement requires an existing document ID in this library.")
        document_id = existing.document_id if existing else (document_id or uuid4())
        text_pages, warnings = _extract(source.suffix.lower(), data)
        bounded_pages = []
        remaining = MAX_EXTRACTED_CHARS
        for page, text in text_pages:
            page_text = text[:remaining]
            if len(page_text) != len(text):
                warnings.append("extracted text truncated at the configured document character bound.")
            remaining -= len(page_text)
            bounded_pages.append((page, page_text))
            if remaining <= 0:
                break
        passages = _make_passages(document_id, bounded_pages)
        if not passages:
            raise LiteratureLibraryError("no extractable text was found; source was not indexed.")
        doc_meta = metadata or LibraryDocumentMetadata()
        title = (doc_meta.title or source.stem).strip()[:500]
        relative = f"sources/{document_id}/{_safe_filename(source.name)}" if retain_original else None
        document = LibraryDocument(
            document_id=document_id,
            library_id=library.library_id,
            display_title=title,
            source_filename=_safe_filename(source.name),
            metadata=doc_meta,
            imported_at_utc=_utc(),
            content_sha256=digest,
            document_revision=(existing.document_revision + 1 if existing else 1),
            extraction_status="indexed",
            extraction_warnings=tuple(dict.fromkeys(warnings)),
            assistant_visible=assistant_visible,
            source_retained=retain_original,
            source_relative_path=relative,
            source_path_private=None if retain_original else str(source.resolve()),
            provenance=(
                "researcher_imported_local_document",
                f"extractor:{source.suffix.lower()}-text-v1",
                f"index:{INDEX_VERSION}",
            ) + ((f"replaced_sha256:{existing.content_sha256}",) if existing else ()),
        )
        document_dir = self._document_dir(library.library_id, document.document_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(document_dir / "record.json", document.model_dump(mode="json"))
        _atomic_json(document_dir / "passages.json", [item.model_dump(mode="json") for item in passages])
        if retain_original:
            destination = self._safe_library_path(library, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not existing:
                raise LiteratureLibraryError("source destination collision; refusing overwrite.")
            _atomic_bytes(destination, data)
        elif existing and existing.source_retained:
            old_source_dir = self._library_dir(library.library_id) / "sources" / str(document_id)
            if old_source_dir.is_dir():
                shutil.rmtree(old_source_dir)
        ids = [item for item in library.document_ids if not existing or item != existing.document_id]
        ids.append(document.document_id)
        self._update_library(library.model_copy(update={"document_ids": tuple(ids)}))
        return document

    def import_literature_record(
        self,
        library_id: UUID | str,
        record,
        *,
        include_abstract: bool = False,
        assistant_visible: bool = False,
        access_status: Literal["public", "private", "unknown"] = "unknown",
    ) -> LibraryDocument:
        """Import an explicitly selected Task 20 metadata record; no network call occurs here."""
        from .literature import ContentLevel, LiteratureRecord, PeerReviewStatus

        if not isinstance(record, LiteratureRecord):
            raise TypeError("record must be an explicitly selected LiteratureRecord.")
        if include_abstract and not record.abstract:
            raise LiteratureLibraryError("the selected literature record has no abstract to import.")
        if include_abstract and record.content_level is ContentLevel.METADATA_ONLY:
            raise LiteratureLibraryError("the literature record does not identify its abstract as available content.")
        library = self.get_library(library_id)
        if library.knowledge_pack_id is not None:
            raise LiteratureLibraryError("Knowledge Pack contents are immutable; create a separate library for additional literature records.")
        doi_identifier = next((item for item in record.identifiers if item.scheme.lower() == "doi"), None)
        linked_doi = next((item.identifier.value for item in record.linked_publication_versions if item.identifier.scheme.lower() == "doi"), None)
        if record.peer_review_status is PeerReviewStatus.PREPRINT:
            publication_status = "preprint"
        elif record.peer_review_status is PeerReviewStatus.PEER_REVIEWED:
            publication_status = "peer_reviewed"
        else:
            publication_status = "unverified"
        source_type = "preprint" if record.work_type.value == "preprint" else "journal_article" if record.work_type.value == "journal_article" else "other"
        access = "public" if access_status == "unknown" and record.access_status.value == "open_access_full_text_available" else access_status
        metadata = LibraryDocumentMetadata(
            title=record.title,
            doi=doi_identifier.value if doi_identifier else None,
            authors=record.authors,
            publication_date=record.publication_date.isoformat() if record.publication_date else record.publication_date_source_text,
            source_type=source_type,
            publication_status=publication_status,
            access_status=access,
            license=record.license_expression,
            evidence_level="abstract_only" if include_abstract else "metadata_only",
            linked_journal_doi=linked_doi,
            identifier_verification=(
                "crossref_verified" if doi_identifier and doi_identifier.verification.value == "verified_by_crossref"
                else "source_asserted" if doi_identifier and doi_identifier.verification.value == "asserted_by_source"
                else "unverified" if doi_identifier else "not_applicable"
            ),
            identifier_source=doi_identifier.verification_source_id if doi_identifier else None,
            source_record_url=record.provenance.source_url,
            source_retrieved_at_utc=record.provenance.retrieved_at_utc,
            publication_status_source=record.review_status_source_id,
            metadata_provenance=(
                f"literature_provider:{record.provenance.provider_id}:{record.provenance.provider_version}",
                f"source_record:{record.provenance.source_record_id}",
            ),
        )
        document_id = uuid4()
        source_representation = json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha256(source_representation.encode()).hexdigest()
        abstract_pages = [(None, record.abstract or "")] if include_abstract else []
        passages = _make_passages(document_id, abstract_pages) if include_abstract else []
        document = LibraryDocument(
            document_id=document_id,
            library_id=library.library_id,
            display_title=(record.title or record.record_id)[:500],
            source_filename=f"{_safe_filename(record.record_id)}.metadata.json",
            metadata=metadata,
            imported_at_utc=_utc(),
            content_sha256=fingerprint,
            fingerprint_kind="literature_record_json_sha256",
            extraction_status="indexed" if include_abstract else "metadata_only",
            extraction_warnings=tuple(record.warnings),
            assistant_visible=assistant_visible,
            source_retained=False,
            provenance=(
                f"source:{record.provenance.provider_id}:{record.provenance.provider_version}",
                f"record:{record.provenance.source_record_id}",
                f"url:{record.provenance.source_url}",
                f"retrieved:{record.provenance.retrieved_at_utc.isoformat()}",
            ) + (("abstract_explicitly_selected",) if include_abstract else ()),
        )
        document_dir = self._document_dir(library.library_id, document_id)
        document_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(document_dir / "record.json", document.model_dump(mode="json"))
        _atomic_json(document_dir / "passages.json", [item.model_dump(mode="json") for item in passages])
        self._update_library(library.model_copy(update={"document_ids": (*library.document_ids, document_id)}))
        return document

    def remove_document(self, library_id: UUID | str, document_id: UUID | str) -> None:
        library = self.get_library(library_id)
        if library.knowledge_pack_id is not None:
            raise LiteratureLibraryError("Knowledge Pack resources cannot be removed individually; remove the selected pack library as a whole.")
        key = UUID(str(document_id))
        if key not in library.document_ids:
            raise LiteratureLibraryError("document is not a member of the selected library.")
        shutil.rmtree(self._document_dir(library.library_id, key))
        source_dir = self._library_dir(library.library_id) / "sources" / str(key)
        if source_dir.is_dir():
            shutil.rmtree(source_dir)
        self._update_library(library.model_copy(update={"document_ids": tuple(i for i in library.document_ids if i != key)}))

    def set_assistant_visibility(self, library_id: UUID | str, document_id: UUID | str, visible: bool) -> LibraryDocument:
        document = self.get_document(library_id, document_id)
        updated = document.model_copy(update={"assistant_visible": bool(visible)})
        _atomic_json(self._document_path(document.library_id, document.document_id), updated.model_dump(mode="json"))
        return updated

    def update_document_metadata(
        self,
        library_id: UUID | str,
        document_id: UUID | str,
        metadata: LibraryDocumentMetadata,
        *,
        provenance_note: str,
    ) -> LibraryDocument:
        if not provenance_note.strip() or len(provenance_note) > 300:
            raise LiteratureLibraryError("metadata update provenance must explain its explicit source.")
        document = self.get_document(library_id, document_id)
        if document.library_id and self.get_library(document.library_id).knowledge_pack_id is not None:
            raise LiteratureLibraryError("Knowledge Pack metadata is immutable; create a new pack version to change its scientific metadata.")
        updated = document.model_copy(
            update={
                "metadata": metadata.model_copy(
                    update={"metadata_provenance": (*metadata.metadata_provenance, provenance_note)}
                )
            }
        )
        _atomic_json(self._document_path(document.library_id, document.document_id), updated.model_dump(mode="json"))
        return updated

    def get_document(self, library_id: UUID | str, document_id: UUID | str) -> LibraryDocument:
        library = self.get_library(library_id)
        key = UUID(str(document_id))
        if key not in library.document_ids:
            raise LiteratureLibraryError("document is not a member of the selected library.")
        try:
            document = LibraryDocument.model_validate_json(self._document_path(library.library_id, key).read_text(encoding="utf-8"), strict=True)
        except (OSError, ValueError) as exc:
            raise LiteratureLibraryError("document record is damaged or unavailable.") from exc
        if document.source_retained:
            source = self._safe_library_path(library, document.source_relative_path)
            if not source.is_file() or _sha256_file(source) != document.content_sha256:
                document = document.model_copy(update={"extraction_status": "changed_needs_review"})
                _atomic_json(self._document_path(library.library_id, key), document.model_dump(mode="json"))
        elif document.source_path_private:
            source = Path(document.source_path_private)
            if not source.is_file():
                document = document.model_copy(update={"extraction_status": "source_missing"})
                _atomic_json(self._document_path(library.library_id, key), document.model_dump(mode="json"))
            elif _sha256_file(source) != document.content_sha256:
                document = document.model_copy(update={"extraction_status": "changed_needs_review"})
                _atomic_json(self._document_path(library.library_id, key), document.model_dump(mode="json"))
        return document

    def load_passages(self, library_id: UUID | str, document_id: UUID | str) -> tuple[LibraryPassage, ...]:
        document = self.get_document(library_id, document_id)
        path = self._document_dir(document.library_id, document.document_id) / "passages.json"
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            passages = tuple(LibraryPassage.model_validate_json(json.dumps(item), strict=True) for item in values)
        except (OSError, ValueError, TypeError) as exc:
            raise LiteratureLibraryError("document passage index is corrupted; re-import for researcher review.") from exc
        if any(item.document_id != document.document_id for item in passages):
            raise LiteratureLibraryError("passage index refers to a different document identity.")
        return passages

    def search(
        self,
        query: str,
        *,
        selected_library_ids: tuple[UUID | str, ...],
        selected_document_ids: tuple[UUID | str, ...] = (),
        source_types: tuple[str, ...] = (),
        publication_statuses: tuple[str, ...] = (),
        assistant_only: bool = False,
        limit: int = 5,
    ) -> tuple[RetrievedLibraryPassage, ...]:
        if not 1 <= limit <= MAX_RESULTS:
            raise LiteratureLibraryError(f"result limit must be between 1 and {MAX_RESULTS}.")
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise LiteratureLibraryError(f"query must contain 1 to {MAX_QUERY_CHARS} characters.")
        terms = _tokens(query)
        if not terms:
            raise LiteratureLibraryError("query contains no searchable terms.")
        doc_filter = {str(item) for item in selected_document_ids}
        candidates = []
        deadline = time.monotonic() + MAX_SEARCH_SECONDS
        selected = tuple(dict.fromkeys(str(item) for item in selected_library_ids))
        if not selected:
            raise LiteratureLibraryError("select at least one library in the trusted interface before searching.")
        for library_id in selected:
            if time.monotonic() > deadline:
                raise LiteratureLibraryError("local search exceeded its three-second execution bound; narrow the corpus.")
            library = self.get_library(library_id)
            for document_id in library.document_ids:
                if doc_filter and str(document_id) not in doc_filter:
                    continue
                document = self.get_document(library.library_id, document_id)
                if document.extraction_status != "indexed" or assistant_only and not document.assistant_visible:
                    continue
                if source_types and document.metadata.source_type not in source_types:
                    continue
                if publication_statuses and document.metadata.publication_status not in publication_statuses:
                    continue
                try:
                    passages = self.load_passages(library.library_id, document_id)
                except LiteratureLibraryError as exc:
                    raise LiteratureLibraryError(
                        f"document {document.document_id} has a corrupted index; inspect or explicitly re-import it."
                    ) from exc
                for passage in passages:
                    if time.monotonic() > deadline:
                        raise LiteratureLibraryError("local search exceeded its three-second execution bound; narrow the corpus.")
                    candidates.append((str(document.document_id), passage.ordinal, library, document, passage, _tokens(passage.text)))
                    if len(candidates) > MAX_SEARCH_PASSAGES:
                        raise LiteratureLibraryError(
                            f"selected corpus exceeds the bounded search limit ({MAX_SEARCH_PASSAGES} passages); narrow the selected libraries or documents."
                        )
        corpus_size = len(candidates)
        if corpus_size:
            document_frequency = {}
            for row in candidates:
                for term in set(row[5]):
                    document_frequency[term] = document_frequency.get(term, 0) + 1
            average_length = sum(len(row[5]) for row in candidates) / corpus_size
        else:
            document_frequency, average_length = {}, 0.0
        ranked = []
        for row in candidates:
            score = _bm25(terms, row[5], document_frequency, corpus_size, average_length)
            if score > 0:
                ranked.append((score, *row[:5]))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        now = _utc()
        return tuple(
            RetrievedLibraryPassage(
                library_id=row[3].library_id,
                library_name=row[3].name,
                scope=row[3].scope,
                document=row[4],
                passage=row[5],
                score=row[0],
                retrieved_at_utc=now,
                knowledge_pack_id=row[3].knowledge_pack_id,
                knowledge_pack_version=row[3].knowledge_pack_version,
                provenance=(
                    f"library:{row[3].library_id}",
                    f"document:{row[4].document_id}",
                    f"passage:{row[5].passage_id}",
                    f"index:{INDEX_VERSION}",
                    *row[3].provenance,
                    *row[4].provenance,
                    *((f"knowledge_pack:{row[3].knowledge_pack_id}", f"knowledge_pack_version:{row[3].knowledge_pack_version}", f"knowledge_pack_sha256:{row[3].knowledge_pack_content_sha256}") if row[3].knowledge_pack_id else ()),
                ),
            )
            for row in ranked[:limit]
        )

    def export_manifest(self, library_id: UUID | str, destination: str | Path) -> Path:
        library = self.get_library(library_id)
        documents = tuple(
            self.get_document(library.library_id, item).model_copy(
                update={"source_retained": False, "source_relative_path": None, "source_path_private": None, "assistant_visible": False}
            )
            for item in library.document_ids
        )
        portable_library = library.model_copy(update={
            "knowledge_pack_id": None,
            "knowledge_pack_version": None,
            "knowledge_pack_content_sha256": None,
        })
        manifest = LibraryManifest(library=portable_library, documents=documents, exported_at_utc=_utc())
        path = Path(destination)
        if path.exists():
            raise FileExistsError("manifest destination already exists; refusing to overwrite.")
        _atomic_json(path, manifest.model_dump(mode="json"))
        return path

    def export_payload_bundle(self, library_id: UUID | str, destination: str | Path, *, researcher_confirmed: bool) -> Path:
        if researcher_confirmed is not True:
            raise LiteratureLibraryError("document payload export requires explicit authorization confirmation.")
        library = self.get_library(library_id)
        path = Path(destination)
        if path.exists():
            raise FileExistsError("bundle destination already exists; refusing to overwrite.")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        self.export_manifest(library.library_id, manifest_path)
        try:
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(manifest_path, "manifest.json")
                for document in self.list_documents(library.library_id):
                    if document.source_retained and document.source_relative_path:
                        source = self._safe_library_path(library, document.source_relative_path)
                        archive.write(source, f"payloads/{document.document_id}/{document.source_filename}")
        finally:
            manifest_path.unlink(missing_ok=True)
        return path

    def import_manifest(self, source: str | Path) -> ScientificLibrary:
        try:
            value = json.loads(Path(source).read_text(encoding="utf-8"))
            manifest = LibraryManifest.model_validate_json(json.dumps(value), strict=True)
        except (OSError, ValueError, TypeError) as exc:
            raise LiteratureLibraryError("library manifest is invalid or unreadable.") from exc
        if manifest.manifest_schema_version != LIBRARY_SCHEMA_VERSION:
            raise LiteratureLibraryError("library manifest version is not supported.")
        if any(item.source_retained or item.source_relative_path or item.source_path_private for item in manifest.documents):
            raise LiteratureLibraryError("portable metadata manifest must not contain local source paths or payload claims.")
        if any(item.extraction_status != "manifest_only" for item in manifest.documents):
            # Manifest export deliberately omits passage bodies and imports as metadata-only records.
            manifest = manifest.model_copy(update={"documents": tuple(item.model_copy(update={"extraction_status": "manifest_only", "assistant_visible": False}) for item in manifest.documents)})
        if any(str(item.library_id) == str(manifest.library.library_id) for item in self.list_libraries()):
            raise LiteratureLibraryError("library ID already exists; manifest import will not replace an existing library.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._empty = False
        library = manifest.library.model_copy(update={
            "document_ids": tuple(item.document_id for item in manifest.documents),
            "knowledge_pack_id": None,
            "knowledge_pack_version": None,
            "knowledge_pack_content_sha256": None,
        })
        catalog = self._load_catalog()
        catalog["libraries"].append(library.model_dump(mode="json"))
        self._library_dir(library.library_id).mkdir(parents=True, exist_ok=False)
        for document in manifest.documents:
            _atomic_json(self._document_path(library.library_id, document.document_id), document.model_dump(mode="json"))
            _atomic_json(self._document_dir(library.library_id, document.document_id) / "passages.json", [])
        _atomic_json(self.catalog_path, catalog)
        return library

    def _update_library(self, updated: ScientificLibrary) -> None:
        catalog = self._load_catalog()
        catalog["libraries"] = [updated.model_dump(mode="json") if item.get("library_id") == str(updated.library_id) else item for item in catalog["libraries"]]
        _atomic_json(self.catalog_path, catalog)

    def _library_dir(self, library_id: UUID) -> Path:
        return self.root / "libraries" / str(library_id)

    def _document_dir(self, library_id: UUID, document_id: UUID) -> Path:
        return self._library_dir(library_id) / "documents" / str(document_id)

    def _document_path(self, library_id: UUID, document_id: UUID) -> Path:
        return self._document_dir(library_id, document_id) / "record.json"

    def _safe_library_path(self, library: ScientificLibrary, relative: str | None) -> Path:
        if not relative:
            raise LiteratureLibraryError("document has no retained local source.")
        candidate = (self._library_dir(library.library_id) / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            candidate.relative_to(self._library_dir(library.library_id).resolve())
        except ValueError as exc:
            raise LiteratureLibraryError("document path escapes its library root.") from exc
        return candidate


def _extract(suffix: str, data: bytes) -> tuple[list[tuple[int | None, str]], list[str]]:
    warnings: list[str] = []
    if suffix in {".txt", ".md", ".markdown"}:
        text = data.decode("utf-8-sig", errors="replace")
        if "\ufffd" in text:
            warnings.append("invalid UTF-8 bytes were replaced during text extraction.")
        return [(None, text)], warnings
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise LiteratureLibraryError("PDF text extraction requires the optional 'literature' extra (pypdf); no OCR is provided.") from exc
        import io
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise LiteratureLibraryError("encrypted PDFs are not supported; provide an authorized readable copy.")
            pages = []
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text:
                    pages.append((index + 1, text))
            if not pages:
                raise LiteratureLibraryError("PDF contains no extractable text; OCR is not available.")
            warnings.append("PDF layout extraction may not preserve equations, tables, captions, or reading order.")
            return pages, warnings
        except LiteratureLibraryError:
            raise
        except Exception as exc:
            raise LiteratureLibraryError(f"PDF extraction failed: {type(exc).__name__}.") from exc
    raise LiteratureLibraryError("supported document types are PDF, TXT, and Markdown.")


def _make_passages(document_id: UUID, pages: list[tuple[int | None, str]]) -> list[LibraryPassage]:
    passages = []
    ordinal = 0
    offset = 0
    for page, raw in pages:
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        sections: list[tuple[str | None, str, int]] = []
        current_heading = None
        cursor = 0
        for block in re.split(r"\n\s*\n", text):
            start = text.find(block, cursor)
            if start < 0:
                start = cursor
            cursor = start + len(block)
            stripped = block.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current_heading = stripped.lstrip("# ")[:200]
                continue
            for index in range(0, len(stripped), MAX_PASSAGE_CHARS):
                fragment = stripped[index:index + MAX_PASSAGE_CHARS]
                char_start = offset + start + index
                char_stop = char_start + len(fragment)
                passage_id = hashlib.sha256(f"{document_id}|{INDEX_VERSION}|{page}|{ordinal}|{char_start}|{fragment}".encode()).hexdigest()
                sections.append((current_heading, fragment, char_start))
                passages.append(LibraryPassage(passage_id=passage_id, document_id=document_id, page_number=page, section=current_heading, ordinal=ordinal, char_start=char_start, char_stop=char_stop, text=fragment))
                ordinal += 1
        offset += len(text)
    return passages


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _TOKEN.findall(text)]


def _bm25(query: list[str], tokens: list[str], df: dict[str, int], corpus_size: int, average_length: float) -> float:
    # Standard BM25 over selected passages; scores rank this corpus only.
    if not tokens:
        return 0.0
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    length = len(tokens)
    score = 0.0
    k1, b = 1.5, 0.75
    for term in query:
        frequency = tf.get(term, 0)
        if not frequency:
            continue
        document_count = df.get(term, 0)
        idf = math.log(1 + (corpus_size - document_count + 0.5) / (document_count + 0.5))
        norm = 1 - b + b * length / max(average_length, 1.0)
        score += idf * frequency * (k1 + 1) / (frequency + k1 * norm)
    return score


def _safe_filename(name: str) -> str:
    value = Path(name).name.replace("\x00", "")
    value = re.sub(r"[^A-Za-z0-9._ -]", "_", value).strip(" .")[:180]
    if not value or value in {".", ".."}:
        raise LiteratureLibraryError("source filename is invalid.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.parent / f".{path.name}.{uuid4()}.tmp"
    try:
        temporary.write_text(data, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4()}.tmp"
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
