from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from neuroscope.assistant.literature import (
    PREPRINT_WARNING,
    AccessStatus,
    AuthorizedLiteratureClient,
    ContentLevel,
    CrossrefDOIVerifier,
    EuropePMCProvider,
    IdentifierVerification,
    LiteratureConsentDecision,
    LiteratureConsentDenied,
    LiteratureSearchService,
    LiteratureServiceError,
    PeerReviewStatus,
    manual_google_scholar_link,
)
from neuroscope.assistant.tools import AssistantToolService


def _approved(disclosure):
    assert disclosure.project_data_included is False
    assert disclosure.query
    assert "sent to the named external" in disclosure.notice
    return LiteratureConsentDecision(
        approved=True,
        exact_query_reviewed=True,
        no_project_data_or_recordings=True,
        no_unpublished_or_confidential_research=True,
    )


def _europe_payload(result):
    return {"resultList": {"result": [result]}}


def test_europe_pmc_requires_per_query_consent_before_transport_and_sends_query_only():
    calls = []

    def transport(url, timeout):
        calls.append((url, timeout))
        return _europe_payload({"id": "123", "title": "A record"})

    provider = EuropePMCProvider(transport=transport, minimum_interval_s=0)
    service = LiteratureSearchService((provider,))
    with pytest.raises(LiteratureConsentDenied, match="declined"):
        service.search(
            provider.provider_id,
            "hippocampal oscillations",
            consent_prompt=lambda disclosure: LiteratureConsentDecision(
                approved=False,
                exact_query_reviewed=False,
                no_project_data_or_recordings=False,
                no_unpublished_or_confidential_research=False,
            ),
            researcher_id="Researcher",
        )
    assert calls == []

    result = service.search(
        provider.provider_id,
        " hippocampal   oscillations ",
        consent_prompt=_approved,
        researcher_id="Researcher",
    )
    assert len(calls) == 1
    query_params = parse_qs(urlsplit(calls[0][0]).query)
    assert query_params["query"] == ["hippocampal oscillations"]
    assert query_params["pageSize"] == ["10"]
    assert "private_recording" not in calls[0][0]
    assert result.records[0].title == "A record"
    assert result.records[0].provenance.source_url.startswith("https://europepmc.org/")
    assert result.records[0].provenance.retrieved_at_utc.tzinfo is not None


def test_europe_pmc_marks_preprints_prominently_and_distinguishes_abstract_from_full_text():
    provider = EuropePMCProvider(
        transport=lambda _url, _timeout: _europe_payload({
            "id": "PPR12345",
            "source": "PPR",
            "title": "Preprint title",
            "authorString": "Ada A; Ben B",
            "doi": "10.1101/2026.01.02.123456",
            "firstPublicationDate": "2026-01-02",
            "isPreprint": True,
            "abstractText": "Abstract excerpt.",
            "isOpenAccess": True,
            "fullTextUrlList": {"fullTextUrl": [{"url": "https://example.org/fulltext"}]},
            "license": "CC BY 4.0",
            "relatedRecords": [{"doi": "10.1000/journal.1", "type": "published-version"}],
        }),
        minimum_interval_s=0,
    )
    record = LiteratureSearchService((provider,)).search(
        provider.provider_id, "neuroscience", consent_prompt=_approved, researcher_id="Researcher"
    ).records[0]
    assert record.peer_review_status is PeerReviewStatus.PREPRINT
    assert record.work_type.value == "preprint"
    assert record.prominent_warning == PREPRINT_WARNING
    assert PREPRINT_WARNING in record.warnings
    assert record.content_level is ContentLevel.ABSTRACT_ONLY
    assert record.access_status is AccessStatus.OPEN_ACCESS_FULL_TEXT_AVAILABLE
    assert record.full_text_available is True
    assert record.full_text_urls == ("https://example.org/fulltext",)
    assert record.license_expression == "CC BY 4.0"
    assert record.license_status.value == "reported_by_source"
    assert record.identifiers[0].verification is IdentifierVerification.SOURCE_ASSERTED
    assert len(record.linked_publication_versions) == 1
    assert record.linked_publication_versions[0].identifier.value == "10.1000/journal.1"
    assert record.linked_publication_versions[0].scientific_identity_verified is False


def test_non_preprint_journal_article_remains_unverified_without_peer_review_evidence():
    provider = EuropePMCProvider(
        transport=lambda _url, _timeout: _europe_payload({
            "id": "MED123",
            "title": "Journal article",
            "pubTypeList": ["Journal Article"],
            "abstractText": "Abstract only.",
            "isOpenAccess": False,
        }),
        minimum_interval_s=0,
    )
    record = LiteratureSearchService((provider,)).search(
        provider.provider_id, "signal", consent_prompt=_approved, researcher_id="Researcher"
    ).records[0]
    assert record.work_type.value == "journal_article"
    assert record.peer_review_status is PeerReviewStatus.UNVERIFIED
    assert record.review_status_source_id is None
    assert record.content_level is ContentLevel.ABSTRACT_ONLY
    assert record.access_status is AccessStatus.ABSTRACT_ONLY
    assert record.full_text_available is False
    assert record.full_text_urls == ()


def test_missing_bibliographic_values_remain_unknown_and_never_get_fabricated():
    provider = EuropePMCProvider(
        transport=lambda _url, _timeout: _europe_payload({"id": "record-unknown"}),
        minimum_interval_s=0,
    )
    record = LiteratureSearchService((provider,)).search(
        provider.provider_id, "unknown work", consent_prompt=_approved, researcher_id="Researcher"
    ).records[0]
    assert record.title is None
    assert record.authors == ()
    assert record.publication_date is None
    assert record.identifiers == ()
    assert record.content_level is ContentLevel.METADATA_ONLY
    assert record.peer_review_status is PeerReviewStatus.UNVERIFIED
    with pytest.raises(LiteratureServiceError, match="stable source record identifier"):
        from neuroscope.assistant.literature import _europe_pmc_record

        _europe_pmc_record({}, query_hash="a" * 64, retrieved_at=record.provenance.retrieved_at_utc)


def test_crossref_doi_lookup_requires_consent_verifies_identifier_and_preserves_partial_date_and_relations():
    requests = []

    def transport(url, _timeout):
        requests.append(url)
        return {"message": {
            "DOI": "10.5555/ARTICLE.1",
            "title": ["Published record"],
            "author": [{"given": "A", "family": "Author"}],
            "type": "journal-article",
            "published-print": {"date-parts": [[2024]]},
            "URL": "https://doi.org/10.5555/article.1",
            "relation": {"is-preprint-of": [{"id-type": "doi", "id": "10.1101/2024.01.01.123456"}]},
        }}

    verifier = CrossrefDOIVerifier(transport=transport, minimum_interval_s=0)
    client = AuthorizedLiteratureClient((), verifier)
    with pytest.raises(LiteratureConsentDenied):
        client.verify_doi(
            "10.5555/article.1",
            consent_prompt=lambda _disclosure: LiteratureConsentDecision(
                approved=False,
                exact_query_reviewed=False,
                no_project_data_or_recordings=False,
                no_unpublished_or_confidential_research=False,
            ),
            researcher_id="Researcher",
        )
    assert requests == []
    record = client.verify_doi(
        "https://doi.org/10.5555/article.1",
        consent_prompt=_approved,
        researcher_id="Researcher",
    )
    assert len(requests) == 1
    assert record.identifiers[0].verification is IdentifierVerification.VERIFIED_BY_CROSSREF
    assert record.peer_review_status is PeerReviewStatus.UNVERIFIED
    assert record.publication_date is None
    assert record.publication_date_source_text == "2024"
    assert len(record.linked_publication_versions) == 1
    linked = record.linked_publication_versions[0]
    assert linked.relation == "is-preprint-of"
    assert linked.scientific_identity_verified is False
    assert "may differ scientifically" in linked.warning


def test_provider_rejects_tampered_or_wrong_query_authorization():
    provider = EuropePMCProvider(
        transport=lambda _url, _timeout: _europe_payload({"id": "1"}),
        minimum_interval_s=0,
    )
    captured = {}
    service = LiteratureSearchService((provider,))

    def consent(disclosure):
        decision = _approved(disclosure)
        captured["query"] = disclosure.query
        return decision

    service.search(provider.provider_id, "spike sorting", consent_prompt=consent, researcher_id="Researcher")
    # A valid consent for one query cannot authorize a changed query.
    from neuroscope.assistant.literature import _authorization

    authorization = _authorization(provider.provider_id, captured["query"], "Researcher")
    with pytest.raises(LiteratureConsentDenied, match="does not match"):
        provider.search("another query", authorization=authorization, limit=5)


def test_provider_authorization_is_single_use_even_for_an_identical_query():
    provider = EuropePMCProvider(
        transport=lambda _url, _timeout: _europe_payload({"id": "1"}),
        minimum_interval_s=0,
    )
    from neuroscope.assistant.literature import _authorization

    authorization = _authorization(provider.provider_id, "same query", "Researcher")
    provider.search("same query", authorization=authorization, limit=2)
    with pytest.raises(LiteratureConsentDenied, match="already been used"):
        provider.search("same query", authorization=authorization, limit=2)


def test_google_scholar_is_only_a_manual_search_url_and_query_paths_are_rejected():
    link = manual_google_scholar_link("spike sorting single channel")
    assert link.provider_id == "google_scholar_manual"
    assert link.scraping_backend is False
    assert "spike+sorting+single+channel" in link.url
    with pytest.raises(ValueError, match="filesystem paths"):
        manual_google_scholar_link(r"C:\private\recording.mat")


def test_literature_service_does_not_add_model_tools_or_project_context():
    tools = {item.name for item in AssistantToolService(lambda: None).tool_schemas}
    assert tools == {
        "get_current_dataset_summary",
        "get_metadata_status",
        "list_available_operations",
        "describe_operation",
        "list_project_executions",
        "get_execution_summary",
        "prepare_scientific_operation",
        "search_local_scientific_library",
    }
    with pytest.raises(ValidationError):
        # A claimed peer-review status without review evidence is not admissible.
        from neuroscope.assistant.literature import (
            LiteratureProvenance,
            LiteratureRecord,
        )

        LiteratureRecord(
            record_id="x",
            peer_review_status=PeerReviewStatus.PEER_REVIEWED,
            provenance=LiteratureProvenance(
                provider_id="fake",
                provider_version="1",
                source_record_id="x",
                source_url="https://example.org/x",
                retrieved_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                query_sha256="a" * 64,
            ),
        )
