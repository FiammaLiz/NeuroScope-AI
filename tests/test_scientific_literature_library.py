from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication

from neuroscope.assistant.library import (
    LibraryDocumentMetadata,
    LiteratureLibraryError,
    LiteratureLibraryStore,
    _extract,
)
from neuroscope.assistant.literature import (
    ContentLevel,
    IdentifierVerification,
    LiteratureIdentifier,
    LiteratureProvenance,
    LiteratureRecord,
    LiteratureWorkType,
    PeerReviewStatus,
)
from neuroscope.assistant.tools import AssistantToolService
from neuroscope.memory.project_store import ProjectStore
from neuroscope.ui.main_window import MainWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _store(tmp_path: Path):
    store = LiteratureLibraryStore(tmp_path / "library")
    return store, store.create_library("Neuroscience", "personal")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_multiple_library_scopes_selection_and_independent_retrieval(tmp_path):
    store = LiteratureLibraryStore(tmp_path / "catalog")
    core = store.create_library("Core", "core")
    personal = store.create_library("Personal", "personal")
    lab = store.create_library("Lab", "laboratory")
    project = store.create_library("Project", "project")
    source = _write(tmp_path, "note.md", "Spike sorting assigns events to units; detection alone does not sort.")
    documents = [
        store.import_document(library.library_id, source, assistant_visible=True)
        for library in (core, personal, lab, project)
    ]
    assert len({item.document_id for item in documents}) == 4
    store.set_active_libraries((personal.library_id, project.library_id))

    hits = store.search("spike sorting", selected_library_ids=store.active_library_ids())

    assert {hit.library_id for hit in hits} == {personal.library_id, project.library_id}
    assert all(hit.passage.text.startswith("Spike sorting") for hit in hits)
    assert all(hit.document.document_id in {documents[1].document_id, documents[3].document_id} for hit in hits)
    assert store.search("spike sorting", selected_library_ids=(lab.library_id,)) != ()


def test_project_scoped_library_does_not_leak_to_another_project(tmp_path):
    first = ProjectStore(tmp_path / "project-a")
    library = first.literature.create_library("Lab references", "laboratory")
    source = _write(tmp_path, "lab.txt", "Laboratory local protocol for spike sorting.")
    first.literature.import_document(library.library_id, source, assistant_visible=True)
    first.literature.set_active_libraries((library.library_id,))
    reopened = ProjectStore.open_existing(tmp_path / "project-a")
    second = ProjectStore(tmp_path / "project-b")

    assert reopened.literature.active_library_ids() == (str(library.library_id),)
    assert len(reopened.literature.search("laboratory protocol", selected_library_ids=reopened.literature.active_library_ids())) == 1
    assert second.literature.list_libraries() == ()


def test_assistant_visibility_is_explicit_and_filter_is_applied(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "private.md", "Unpublished protocol about adaptive thresholds.")
    document = store.import_document(library.library_id, source)
    store.set_active_libraries((library.library_id,))

    assert store.search("adaptive thresholds", selected_library_ids=(library.library_id,))
    assert store.search("adaptive thresholds", selected_library_ids=(library.library_id,), assistant_only=True) == ()
    store.set_assistant_visibility(library.library_id, document.document_id, True)
    assert store.search("adaptive thresholds", selected_library_ids=(library.library_id,), assistant_only=True)


def test_external_provider_tool_requires_document_level_consent_and_active_selection(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "private.txt", "Private evidence about inter-spike intervals.")
    document = store.import_document(library.library_id, source, assistant_visible=True)
    store.set_active_libraries((library.library_id,))

    def state(allowlist=()):
        return SimpleNamespace(
            literature_store=store,
            active_literature_library_ids=store.active_library_ids(),
            external_provider=True,
            externally_shareable_document_ids=allowlist,
        )

    service = AssistantToolService(lambda: state())
    denied = service.call("search_local_scientific_library", '{"query":"inter-spike intervals"}')
    assert denied.ok and denied.data["passages"] == []
    assert "consent" in denied.summary

    service = AssistantToolService(lambda: state((str(document.document_id),)))
    allowed = service.call("search_local_scientific_library", '{"query":"inter-spike intervals"}')
    assert len(allowed.data["passages"]) == 1
    assert allowed.data["passages"][0]["text"] == "Private evidence about inter-spike intervals."
    assert allowed.data["passages"][0]["passage_id"]


def test_publication_warnings_and_metadata_unknowns_are_preserved(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "preprint.md", "A preprint describes a spike detector.")
    metadata = LibraryDocumentMetadata(
        title="Preprint title",
        source_type="preprint",
        publication_status="preprint",
        access_status="unknown",
        evidence_level="abstract_only",
    )
    document = store.import_document(library.library_id, source, metadata=metadata, assistant_visible=True)
    store.set_active_libraries((library.library_id,))
    service = AssistantToolService(lambda: SimpleNamespace(
        literature_store=store,
        active_literature_library_ids=store.active_library_ids(),
        external_provider=False,
        externally_shareable_document_ids=(),
    ))

    result = service.call("search_local_scientific_library", '{"query":"spike detector"}')

    hit = result.data["passages"][0]
    assert hit["document_id"] == str(document.document_id)
    assert hit["publication_status"] == "preprint"
    assert hit["status_warning"] == "Preprint — not peer reviewed."
    assert hit["evidence_warning"] == "Abstract-level evidence only."
    assert hit["access_status"] == "unknown"


def test_explicit_metadata_status_update_retains_researcher_provenance(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "status.txt", "A literature status fixture.")
    document = store.import_document(library.library_id, source)
    updated_metadata = document.metadata.model_copy(update={
        "access_status": "private",
        "publication_status": "peer_reviewed",
        "publication_status_source": "researcher_entered_not_independently_verified",
    })

    updated = store.update_document_metadata(
        library.library_id,
        document.document_id,
        updated_metadata,
        provenance_note="researcher_explicit_status_selection",
    )

    assert updated.metadata.access_status == "private"
    assert updated.metadata.publication_status == "peer_reviewed"
    assert updated.metadata.publication_status_source == "researcher_entered_not_independently_verified"
    assert updated.metadata.metadata_provenance == ("researcher_explicit_status_selection",)


def test_replacement_requires_explicit_intent_preserves_document_id_and_reindexes(tmp_path):
    store, library = _store(tmp_path)
    original = _write(tmp_path, "draft.txt", "The first version discusses spike detection.")
    document = store.import_document(library.library_id, original)
    replacement = _write(tmp_path, "revised.txt", "The revised version discusses spike sorting.")

    with pytest.raises(LiteratureLibraryError, match="explicit replace_existing"):
        store.import_document(library.library_id, replacement, document_id=document.document_id)
    revised = store.import_document(
        library.library_id, replacement, document_id=document.document_id, replace_existing=True
    )

    assert revised.document_id == document.document_id
    assert revised.document_revision == document.document_revision + 1
    assert revised.content_sha256 != document.content_sha256
    assert not store.search("detection", selected_library_ids=(library.library_id,))
    assert store.search("spike sorting", selected_library_ids=(library.library_id,))


def test_metadata_manifest_round_trip_excludes_text_and_source_paths(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "reference.txt", "Confidential complete source passage.")
    document = store.import_document(library.library_id, source, retain_original=True, assistant_visible=True)
    manifest = store.export_manifest(library.library_id, tmp_path / "manifest.json")
    raw = manifest.read_text(encoding="utf-8")
    assert "Confidential complete source passage" not in raw
    assert str(tmp_path) not in raw

    imported = LiteratureLibraryStore(tmp_path / "other-catalog")
    restored = imported.import_manifest(manifest)
    imported_doc = imported.list_documents(restored.library_id)[0]
    assert imported_doc.document_id == document.document_id
    assert imported_doc.content_sha256 == document.content_sha256
    assert imported_doc.extraction_status == "manifest_only"
    assert imported_doc.assistant_visible is False
    assert imported.search("confidential passage", selected_library_ids=(restored.library_id,)) == ()


def test_payload_export_requires_explicit_authorization_and_refuses_overwrite(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "authorized.txt", "Authorized article text.")
    store.import_document(library.library_id, source, retain_original=True)
    destination = tmp_path / "corpus.zip"
    with pytest.raises(LiteratureLibraryError, match="explicit authorization"):
        store.export_payload_bundle(library.library_id, destination, researcher_confirmed=False)
    store.export_payload_bundle(library.library_id, destination, researcher_confirmed=True)
    with pytest.raises(FileExistsError):
        store.export_payload_bundle(library.library_id, destination, researcher_confirmed=True)


def test_reopen_reuses_saved_index_changed_source_and_corruption_are_reported(tmp_path, monkeypatch):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "saved.md", "Neural activity and MUA bandwidth.")
    document = store.import_document(library.library_id, source, retain_original=True)
    reopened = LiteratureLibraryStore(tmp_path / "library", create=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unchanged documents must not be extracted again")

    monkeypatch.setattr("neuroscope.assistant.library._extract", forbidden)
    assert reopened.search("MUA bandwidth", selected_library_ids=(library.library_id,))
    assert reopened.validate_existing()[0]["status"] == "available"
    library_record = reopened.get_library(library.library_id)
    retained = reopened._safe_library_path(library_record, document.source_relative_path)
    retained.write_text("Changed content requiring review.", encoding="utf-8")
    assert reopened.get_document(library.library_id, document.document_id).extraction_status == "changed_needs_review"
    monkeypatch.undo()
    second_source = _write(tmp_path, "second.txt", "A separate searchable entry.")
    second = reopened.import_document(library.library_id, second_source)
    index = reopened._document_dir(library.library_id, second.document_id) / "passages.json"
    index.write_text("not json", encoding="utf-8")
    with pytest.raises(LiteratureLibraryError, match="corrupted index"):
        reopened.search("separate entry", selected_library_ids=(library.library_id,))


def test_manifest_rejects_duplicate_or_path_claims_and_invalid_queries(tmp_path):
    store, library = _store(tmp_path)
    source = _write(tmp_path, "small.txt", "Neural signal analysis.")
    doc = store.import_document(library.library_id, source)
    manifest = store.export_manifest(library.library_id, tmp_path / "manifest.json")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["documents"][0]["source_relative_path"] = "../../outside.pdf"
    value["documents"][0]["source_retained"] = True
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps(value), encoding="utf-8")
    target = LiteratureLibraryStore(tmp_path / "target")
    with pytest.raises(LiteratureLibraryError):
        target.import_manifest(bad)
    with pytest.raises(LiteratureLibraryError, match="query"):
        store.search(" ", selected_library_ids=(library.library_id,))
    assert doc.document_id in store.get_library(library.library_id).document_ids


def test_pdf_extraction_reports_optional_parser_requirement_when_unavailable():
    import importlib.util

    if importlib.util.find_spec("pypdf") is not None:
        pytest.skip("pypdf is installed; real PDF parser behavior needs a PDF fixture.")
    with pytest.raises(LiteratureLibraryError, match="optional 'literature' extra.*no OCR"):
        _extract(".pdf", b"%PDF-1.4 synthetic placeholder")


def test_project_store_reopens_old_projects_without_creating_libraries(tmp_path):
    project = tmp_path / "legacy"; project.mkdir()
    store = ProjectStore.open_existing(project)
    assert store.literature.list_libraries() == ()
    assert not (project / "literature").exists()


def test_selected_literature_metadata_and_abstract_import_preserves_provenance(tmp_path):
    store, library = _store(tmp_path)
    record = LiteratureRecord(
        record_id="fixture-abstract-1",
        title="Synthetic evaluation record",
        authors=("Evaluation Fixture",),
        work_type=LiteratureWorkType.PREPRINT,
        peer_review_status=PeerReviewStatus.PREPRINT,
        identifiers=(LiteratureIdentifier(scheme="doi", value="10.5555/synthetic.fixture", verification=IdentifierVerification.SOURCE_ASSERTED),),
        provenance=LiteratureProvenance(
            provider_id="fixture",
            provider_version="fixture-v1",
            source_record_id="fixture-abstract-1",
            source_url="https://example.invalid/fixture",
            retrieved_at_utc=datetime.now(timezone.utc),
            query_sha256="a" * 64,
        ),
        abstract="Synthetic abstract about multi-unit activity without sorting.",
        content_level=ContentLevel.ABSTRACT_ONLY,
        access_status="abstract_only",
    )

    document = store.import_literature_record(
        library.library_id, record, include_abstract=True, assistant_visible=True
    )

    assert document.metadata.doi == "10.5555/synthetic.fixture"
    assert document.metadata.identifier_verification == "source_asserted"
    assert document.metadata.publication_status == "preprint"
    assert document.metadata.evidence_level == "abstract_only"
    assert document.fingerprint_kind == "literature_record_json_sha256"
    assert document.provenance[0] == "source:fixture:fixture-v1"
    hits = store.search("multi-unit activity", selected_library_ids=(library.library_id,), assistant_only=True)
    assert hits[0].passage.text == record.abstract


def test_controlled_evaluation_suite_covers_policy_and_neuroscience_cases():
    suite = json.loads(Path("examples/literature_evaluation_cases.json").read_text(encoding="utf-8"))
    ids = {item["case_id"] for item in suite["cases"]}
    assert suite["fixture_kind"].startswith("synthetic_")
    assert suite["model_quality_claim"] is False
    assert len(ids) == len(suite["cases"]) and len(ids) >= 20
    for required in (
        "detection-vs-sorting", "single-channel-sorting", "mua-without-sorting",
        "noise-threshold", "isi-limitations", "mua-bandwidth", "correlation-causation",
        "document-injection", "external-consent", "changed-document", "no-execution-expansion",
    ):
        assert required in ids


def test_literature_workspace_selects_corpus_and_displays_exact_passages(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.store = ProjectStore(tmp_path / "project")
    window._literature_store = window.store.literature
    library = window._literature_store.create_library("Lab notes", "laboratory")
    source = _write(tmp_path, "visible.md", "## Spike sorting\n\nDetection and sorting are different operations.")
    document = window._literature_store.import_document(
        library.library_id, source, assistant_visible=True
    )
    window._refresh_literature_libraries()
    assert "Literature" in [window.workspace_tabs.tabText(i) for i in range(window.workspace_tabs.count())]
    assert window.knowledge_pack_export_button.text().startswith("Create / export Knowledge Pack")
    assert window.knowledge_pack_import_button.text() == "Inspect / import Knowledge Pack"
    assert window.knowledge_pack_inspect_button.text() == "Inspect selected Knowledge Pack"
    assert window.literature_documents.selectionMode() == QAbstractItemView.ExtendedSelection
    window.literature_library_box.item(0).setCheckState(Qt.Checked)
    assert window._literature_store.active_library_ids() == (str(library.library_id),)
    window.literature_query_input.setText("spike sorting")
    window._search_literature_library()
    assert "Detection and sorting are different operations." in window.literature_results.toPlainText()
    assert str(document.document_id) in window.literature_results.toPlainText()
    window._assistant_external_document_allowlist = (str(document.document_id),)
    window.assistant_provider_box.setCurrentIndex(window.assistant_provider_box.findData("ollama"))
    assert window._assistant_external_document_allowlist == ()
    window.close()
    app.processEvents()

