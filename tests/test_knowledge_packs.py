from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from neuroscope.assistant.knowledge_packs import (
    KnowledgePackError,
    KnowledgePackManifest,
    KnowledgePackResource,
    KnowledgePackService,
)
from neuroscope.assistant.library import LibraryDocumentMetadata, LiteratureLibraryStore
from neuroscope.assistant.tools import AssistantToolService
from neuroscope.memory.project_store import ProjectStore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _store(tmp_path: Path):
    store = LiteratureLibraryStore(tmp_path / "catalog")
    library = store.create_library("Researcher library", "laboratory")
    source = tmp_path / "methods.md"
    source.write_text("Negative-going spike detection uses a signed threshold.", encoding="utf-8")
    doc = store.import_document(
        library.library_id,
        source,
        metadata=LibraryDocumentMetadata(
            title="Spike detection methods",
            doi="10.1000/example",
            source_type="preprint",
            publication_status="preprint",
            access_status="private",
            license=None,
            evidence_level="full_text",
        ),
        assistant_visible=True,
        retain_original=True,
    )
    return store, library, doc


def _create_reference_pack(store, library, doc, path):
    service = KnowledgePackService(store)
    return service.create_from_library(
        library.library_id,
        (doc.document_id,),
        path,
        name="Spike methods",
        version="1.2.0",
        pack_id=uuid4(),
        scientific_domains=("computational neuroscience",),
    )


def test_reference_only_pack_round_trip_is_inactive_and_searchable_after_explicit_activation(tmp_path):
    store, library, source_doc = _store(tmp_path)
    pack_path = _create_reference_pack(store, library, source_doc, tmp_path / "methods.nskpack")
    service = KnowledgePackService(store)
    preview = service.inspect(pack_path)

    assert preview.manifest.version == "1.2.0"
    assert preview.manifest.resources[0].metadata.license is None
    assert preview.manifest.resources[0].payload_path is None
    assert preview.manifest.privacy_classification == "private"
    assert preview.payload_count == 0
    assert "Preprint — not peer reviewed." in preview.warnings
    assert any("license is unknown" in warning for warning in preview.warnings)

    target = LiteratureLibraryStore(tmp_path / "target")
    installed = KnowledgePackService(target).import_pack(pack_path)
    installed_doc = target.list_documents(installed.library_id)[0]
    assert installed.knowledge_pack_id == preview.manifest.pack_id
    assert installed.knowledge_pack_version == "1.2.0"
    assert target.active_library_ids() == ()
    assert installed_doc.assistant_visible is False
    assert installed_doc.metadata.evidence_level == "metadata_only"
    assert target.search("signed threshold", selected_library_ids=(installed.library_id,)) == ()
    target.set_assistant_visibility(installed.library_id, installed_doc.document_id, True)
    KnowledgePackService(target).set_active(installed.knowledge_pack_id, "1.2.0", True)
    # Reference-only packs intentionally contain no searchable document text.
    assert target.search("signed threshold", selected_library_ids=target.active_library_ids(), assistant_only=True) == ()


def test_payload_export_requires_both_explicit_authorizations_and_preserves_source(tmp_path):
    store, library, document = _store(tmp_path)
    service = KnowledgePackService(store)
    with pytest.raises(KnowledgePackError, match="redistribution"):
        service.create_from_library(
            library.library_id, (document.document_id,), tmp_path / "blocked.nskpack",
            name="Private pack", version="1.0.0", pack_id=uuid4(),
            include_document_payloads=True,
        )
    with pytest.raises(KnowledgePackError, match="private or access-unknown"):
        service.create_from_library(
            library.library_id, (document.document_id,), tmp_path / "blocked-private.nskpack",
            name="Private pack", version="1.0.0", pack_id=uuid4(),
            include_document_payloads=True, researcher_confirmed_redistribution=True,
        )
    path = service.create_from_library(
        library.library_id,
        (document.document_id,),
        tmp_path / "authorized.nskpack",
        name="Private pack",
        version="1.0.0",
        pack_id=uuid4(),
        include_document_payloads=True,
        researcher_confirmed_redistribution=True,
        researcher_confirmed_private_payloads=True,
    )
    preview = service.inspect(path)
    assert preview.payload_count == 1
    assert preview.manifest.redistribution_status == "researcher_confirmed"
    assert preview.manifest.resources[0].metadata.license is None
    target = LiteratureLibraryStore(tmp_path / "target")
    installed = KnowledgePackService(target).import_pack(path)
    imported_doc = target.list_documents(installed.library_id)[0]
    assert imported_doc.source_retained is True
    assert imported_doc.assistant_visible is False
    source_path = target._safe_library_path(installed, imported_doc.source_relative_path)
    assert source_path.read_text(encoding="utf-8").startswith("Negative-going")


def test_authored_methodology_resource_is_packaged_and_indexed(tmp_path):
    store, library, _doc = _store(tmp_path)
    resource = KnowledgePackResource(
        resource_id=uuid4(),
        kind="methodology",
        title="Local detector guidance",
        authored_content="A median-centered MAD estimator differs from median absolute amplitude.",
    )
    path = KnowledgePackService(store).create_from_library(
        library.library_id, (), tmp_path / "guide.nskpack",
        name="Detector guide", version="1.0.0", pack_id=uuid4(), supplements=(resource,),
    )
    target = LiteratureLibraryStore(tmp_path / "target")
    installed = KnowledgePackService(target).import_pack(path)
    doc = target.list_documents(installed.library_id)[0]
    target.set_assistant_visibility(installed.library_id, doc.document_id, True)
    hits = target.search("median absolute amplitude", selected_library_ids=(installed.library_id,), assistant_only=True)
    assert hits[0].passage.text == resource.authored_content
    assert f"pack_resource_id:{resource.resource_id}" in hits[0].provenance


@pytest.mark.parametrize("mutator, message", [
    (lambda obj: obj.update(schema_version=999), "schema .* unsupported"),
    (lambda obj: obj.update(version="bad"), "malformed|schema"),
])
def test_rejects_unsupported_or_malformed_manifest(tmp_path, mutator, message):
    store, library, doc = _store(tmp_path)
    path = _create_reference_pack(store, library, doc, tmp_path / "good.nskpack")
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    mutator(manifest)
    bad = tmp_path / "bad.nskpack"
    with zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, content in entries.items():
            if name != "manifest.json":
                archive.writestr(name, content)
    with pytest.raises(KnowledgePackError, match=message):
        KnowledgePackService(LiteratureLibraryStore(tmp_path / "target")).inspect(bad)


def test_rejects_path_traversal_and_unreferenced_archive_entries(tmp_path):
    store, library, doc = _store(tmp_path)
    path = _create_reference_pack(store, library, doc, tmp_path / "good.nskpack")
    malicious = tmp_path / "traversal.nskpack"
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json")
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("../outside.txt", b"must not be extracted")
    with pytest.raises(KnowledgePackError, match="unsafe archive path"):
        KnowledgePackService(store).inspect(malicious)
    extra = tmp_path / "extra.nskpack"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(extra, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("payloads/unreferenced.txt", b"data")
    with pytest.raises(KnowledgePackError, match="unreferenced or missing"):
        KnowledgePackService(store).inspect(extra)


def test_changed_payload_and_manifest_content_are_detected(tmp_path):
    store, library, doc = _store(tmp_path)
    path = KnowledgePackService(store).create_from_library(
        library.library_id, (doc.document_id,), tmp_path / "payload.nskpack",
        name="Payload pack", version="1.0.0", pack_id=uuid4(),
        include_document_payloads=True,
        researcher_confirmed_redistribution=True,
        researcher_confirmed_private_payloads=True,
    )
    changed = tmp_path / "changed.nskpack"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(changed, "w", zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            content = source.read(name)
            if name.startswith("payloads/"):
                content += b"tampered"
            target.writestr(name, content)
    with pytest.raises(KnowledgePackError, match="fingerprint mismatch"):
        KnowledgePackService(store).inspect(changed)


def test_duplicate_pack_version_never_overwrites_and_removal_is_isolated(tmp_path):
    store, library, doc = _store(tmp_path)
    service = KnowledgePackService(store)
    first_path = _create_reference_pack(store, library, doc, tmp_path / "one.nskpack")
    installed = service.import_pack(first_path)
    second_source = tmp_path / "other.txt"
    second_source.write_text("An unrelated library remains present.", encoding="utf-8")
    other_library = store.create_library("Unrelated", "personal")
    store.import_document(other_library.library_id, second_source)
    with pytest.raises(KnowledgePackError, match="already installed"):
        service.import_pack(first_path)
    service.set_active(installed.knowledge_pack_id, installed.knowledge_pack_version, True)
    service.remove(installed.knowledge_pack_id, installed.knowledge_pack_version)
    assert store.get_library(other_library.library_id).name == "Unrelated"
    assert store.active_library_ids() == ()


def test_pack_retrieval_tool_returns_pack_version_and_source_provenance(tmp_path):
    store, library, doc = _store(tmp_path)
    path = KnowledgePackService(store).create_from_library(
        library.library_id, (doc.document_id,), tmp_path / "pack.nskpack",
        name="Pack content", version="1.2.0", pack_id=uuid4(),
        include_document_payloads=True,
        researcher_confirmed_redistribution=True,
        researcher_confirmed_private_payloads=True,
    )
    pack = KnowledgePackService(store).import_pack(path, activate=True)
    pack_doc = store.list_documents(pack.library_id)[0]
    store.set_assistant_visibility(pack.library_id, pack_doc.document_id, True)
    state = SimpleNamespace(
        literature_store=store,
        active_literature_library_ids=store.active_library_ids(),
        external_provider=False,
        externally_shareable_document_ids=(),
    )
    result = AssistantToolService(lambda: state).call(
        "search_local_scientific_library", '{"query":"signed threshold"}'
    )
    hit = result.data["passages"][0]
    assert hit["knowledge_pack_id"] == str(pack.knowledge_pack_id)
    assert hit["knowledge_pack_version"] == "1.2.0"
    assert hit["content_fingerprint"] == store.list_documents(pack.library_id)[0].content_sha256
    assert f"knowledge_pack:{pack.knowledge_pack_id}" in hit["provenance"]
    assert f"pack_resource_id:{doc.document_id}" in hit["provenance"]
    assert hit["status_warning"] == "Preprint — not peer reviewed."
    external_without_consent = SimpleNamespace(
        literature_store=store,
        active_literature_library_ids=store.active_library_ids(),
        external_provider=True,
        externally_shareable_document_ids=(),
    )
    denied = AssistantToolService(lambda: external_without_consent).call(
        "search_local_scientific_library", '{"query":"signed threshold"}'
    )
    assert denied.data["passages"] == []
    assert denied.data["external_sharing"] == "not_authorized"


def test_imported_manifest_is_offline_and_project_scoped(tmp_path, monkeypatch):
    store, library, doc = _store(tmp_path)
    path = _create_reference_pack(store, library, doc, tmp_path / "pack.nskpack")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("pack import attempted network access"))
    first_project = ProjectStore(tmp_path / "project-one")
    imported = first_project.knowledge_packs.import_pack(path, activate=False)
    second_project = ProjectStore(tmp_path / "project-two")
    assert imported.library_id in {item.library_id for item in first_project.literature.list_libraries()}
    assert second_project.literature.list_libraries() == ()
    reopened = ProjectStore.open_existing(tmp_path / "project-one")
    assert reopened.literature.get_library(imported.library_id).knowledge_pack_content_sha256 == imported.knowledge_pack_content_sha256
    assert reopened.literature.active_library_ids() == ()


def test_pack_models_reject_duplicate_resources_and_unsafe_payload_paths():
    resource_id = uuid4()
    base = {
        "resource_id": resource_id,
        "kind": "document",
        "title": "Bad path",
        "payload_path": "payloads/../outside.txt",
        "payload_sha256": hashlib.sha256(b"x").hexdigest(),
        "payload_filename": "outside.txt",
    }
    with pytest.raises(ValueError, match="safe package-relative"):
        KnowledgePackResource.model_validate(base)


def test_library_activation_can_be_disabled_without_removing_documents(tmp_path):
    store, library, doc = _store(tmp_path)
    path = _create_reference_pack(store, library, doc, tmp_path / "pack.nskpack")
    service = KnowledgePackService(store)
    imported = service.import_pack(path, activate=True)
    assert store.active_library_ids() == (str(imported.library_id),)
    service.set_active(imported.knowledge_pack_id, imported.knowledge_pack_version, False)
    assert store.active_library_ids() == ()
    assert store.list_documents(imported.library_id)


def test_installed_pack_contents_are_immutable_but_visibility_remains_local(tmp_path):
    store, library, doc = _store(tmp_path)
    path = _create_reference_pack(store, library, doc, tmp_path / "pack.nskpack")
    installed = KnowledgePackService(store).import_pack(path)
    installed_doc = store.list_documents(installed.library_id)[0]
    source = tmp_path / "replacement.md"
    source.write_text("Replacement should not mutate a versioned pack.", encoding="utf-8")
    with pytest.raises(ValueError, match="Knowledge Pack contents are immutable"):
        store.import_document(installed.library_id, source, document_id=installed_doc.document_id, replace_existing=True)
    with pytest.raises(ValueError, match="cannot be removed individually"):
        store.remove_document(installed.library_id, installed_doc.document_id)
    with pytest.raises(ValueError, match="metadata is immutable"):
        store.update_document_metadata(
            installed.library_id, installed_doc.document_id,
            installed_doc.metadata.model_copy(update={"access_status": "public"}),
            provenance_note="test status update",
        )
    visible = store.set_assistant_visibility(installed.library_id, installed_doc.document_id, True)
    assert visible.assistant_visible is True
    assert KnowledgePackService(store).installed_manifest(installed.knowledge_pack_id, installed.knowledge_pack_version).package_sha256 == installed.knowledge_pack_content_sha256


def test_manifest_package_hash_is_stable_and_declared_provenance_survives(tmp_path):
    store, library, doc = _store(tmp_path)
    pack_path = _create_reference_pack(store, library, doc, tmp_path / "one.nskpack")
    preview = KnowledgePackService(store).inspect(pack_path)
    assert len(preview.manifest.package_sha256) == 64
    assert preview.manifest.resources[0].source_document_id == doc.document_id
    assert preview.manifest.resources[0].source_library_id == library.library_id
    assert preview.manifest.resources[0].content_fingerprint == doc.content_sha256


def test_same_pack_id_and_version_cannot_be_reused_for_different_contents(tmp_path):
    store, library, doc = _store(tmp_path)
    pack_id = uuid4()
    service = KnowledgePackService(store)
    service.create_from_library(
        library.library_id, (doc.document_id,), tmp_path / "v1.nskpack",
        name="Immutable", version="1.0.0", pack_id=pack_id,
    )
    second = tmp_path / "second.txt"
    second.write_text("Different scientific note.", encoding="utf-8")
    other = store.import_document(library.library_id, second)
    with pytest.raises(KnowledgePackError, match="immutable"):
        service.create_from_library(
            library.library_id, (other.document_id,), tmp_path / "v1-changed.nskpack",
            name="Immutable", version="1.0.0", pack_id=pack_id,
        )


def test_unsupported_application_compatibility_is_visible_and_import_rejected(tmp_path):
    store, library, doc = _store(tmp_path)
    path = KnowledgePackService(store).create_from_library(
        library.library_id, (doc.document_id,), tmp_path / "incompatible.nskpack",
        name="Future pack", version="1.0.0", pack_id=uuid4(),
        minimum_neuroscope_version="99.0.0",
    )
    service = KnowledgePackService(LiteratureLibraryStore(tmp_path / "target"))
    preview = service.inspect(path)
    assert any("incompatible" in item for item in preview.warnings)
    with pytest.raises(KnowledgePackError, match="not compatible"):
        service.import_pack(path)


def test_reopening_installed_pack_reuses_passages_without_extraction(tmp_path, monkeypatch):
    source_store, source_library, source_doc = _store(tmp_path)
    path = KnowledgePackService(source_store).create_from_library(
        source_library.library_id, (source_doc.document_id,), tmp_path / "pack.nskpack",
        name="Indexed pack", version="1.0.0", pack_id=uuid4(),
        include_document_payloads=True,
        researcher_confirmed_redistribution=True,
        researcher_confirmed_private_payloads=True,
    )
    project = ProjectStore(tmp_path / "pack-project")
    installed = project.knowledge_packs.import_pack(path, activate=True)
    installed_doc = project.literature.list_documents(installed.library_id)[0]
    project.literature.set_assistant_visibility(installed.library_id, installed_doc.document_id, True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reopening an unchanged pack must not extract or rebuild passages")

    monkeypatch.setattr("neuroscope.assistant.library._extract", forbidden)
    reopened = ProjectStore.open_existing(tmp_path / "pack-project")
    reopened_pack = reopened.literature.get_library(installed.library_id)
    manifest = reopened.knowledge_packs.installed_manifest(reopened_pack.knowledge_pack_id, reopened_pack.knowledge_pack_version)
    hits = reopened.literature.search("signed threshold", selected_library_ids=reopened.literature.active_library_ids(), assistant_only=True)
    assert manifest.package_sha256 == installed.knowledge_pack_content_sha256
    assert len(hits) == 1


def test_changed_installed_pack_manifest_is_reported_without_blocking_project_open(tmp_path):
    source_store, source_library, source_doc = _store(tmp_path)
    pack_path = _create_reference_pack(source_store, source_library, source_doc, tmp_path / "pack.nskpack")
    project = ProjectStore(tmp_path / "project")
    installed = project.knowledge_packs.import_pack(pack_path)
    manifest_path = project.literature._library_dir(installed.library_id) / "knowledge_pack.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    reopened = ProjectStore.open_existing(tmp_path / "project")

    assert reopened.literature_recovery_status.startswith("unavailable:")
    assert reopened.manifest is not None


def test_duplicate_resource_ids_are_rejected_by_manifest():
    resource_id = uuid4()
    resource = KnowledgePackResource(resource_id=resource_id, kind="reference", title="One")
    with pytest.raises(ValueError, match="resource identifiers must be unique"):
        KnowledgePackManifest(
            pack_id=uuid4(), name="Duplicate", version="1.0.0", created_at_utc="2026-01-01T00:00:00Z",
            modified_at_utc="2026-01-01T00:00:00Z", resources=(resource, resource), package_sha256="0" * 64,
        )

