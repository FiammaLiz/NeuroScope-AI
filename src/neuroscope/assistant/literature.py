"""Consent-gated, read-only literature metadata interfaces.

No provider is wired into assistant tools. These adapters send only a researcher-
entered literature query or DOI after an explicit, per-request consent callback.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
import urllib.error
import urllib.parse
from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from enum import StrEnum
from threading import Lock
from typing import Any, Literal, Protocol
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from ..analysis.contracts import OperationModel

PREPRINT_WARNING = "Preprint — not peer reviewed."
EUROPE_PMC_ID = "europe_pmc"
CROSSREF_ID = "crossref"
_AUTHORIZATION_SECRET = secrets.token_bytes(32)
_AUTHORIZATION_LOCK = Lock()
_ISSUED_AUTHORIZATIONS: set[str] = set()


class PeerReviewStatus(StrEnum):
    PEER_REVIEWED = "peer_reviewed_journal_article"
    PREPRINT = "preprint_not_peer_reviewed"
    UNVERIFIED = "review_status_unverified"


class LiteratureWorkType(StrEnum):
    JOURNAL_ARTICLE = "journal_article"
    PREPRINT = "preprint"
    OTHER = "other"
    UNVERIFIED = "work_type_unverified"


class ContentLevel(StrEnum):
    METADATA_ONLY = "metadata_only"
    ABSTRACT_ONLY = "abstract_only"
    FULL_TEXT = "full_text"


class AccessStatus(StrEnum):
    OPEN_ACCESS_FULL_TEXT_AVAILABLE = "open_access_full_text_available"
    ABSTRACT_ONLY = "abstract_only"
    RESTRICTED_OR_UNKNOWN = "restricted_or_unknown"


class LicenseStatus(StrEnum):
    REPORTED_BY_SOURCE = "reported_by_source"
    NOT_REPORTED = "not_reported"


class IdentifierVerification(StrEnum):
    VERIFIED_BY_CROSSREF = "verified_by_crossref"
    SOURCE_ASSERTED = "asserted_by_source"
    UNVERIFIED = "unverified"


class LiteratureIdentifier(OperationModel):
    scheme: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=300)
    verification: IdentifierVerification = IdentifierVerification.UNVERIFIED
    verification_source_id: str | None = None
    verified_at_utc: datetime | None = None

    @model_validator(mode="after")
    def verification_provenance(self):
        if (
            self.verification is IdentifierVerification.VERIFIED_BY_CROSSREF
            and (self.verification_source_id != CROSSREF_ID or self.verified_at_utc is None)
        ):
            raise ValueError("Crossref-verified identifiers require Crossref provenance and a UTC verification time.")
        return self


class LiteratureProvenance(OperationModel):
    provider_id: str = Field(min_length=1, max_length=80)
    provider_version: str = Field(min_length=1, max_length=80)
    source_record_id: str = Field(min_length=1, max_length=300)
    source_url: str = Field(pattern=r"^https://")
    retrieved_at_utc: datetime
    query_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("retrieved_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("retrieved_at_utc must be timezone-aware and expressed in UTC.")
        return value


class LinkedPublicationVersion(OperationModel):
    identifier: LiteratureIdentifier
    relation: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=80)
    source_url: str = Field(pattern=r"^https://")
    scientific_identity_verified: bool = False
    warning: str = "A linked version may differ scientifically; compare its content and methods directly."


class LiteratureRecord(OperationModel):
    record_id: str = Field(min_length=1, max_length=300)
    title: str | None = Field(default=None, max_length=2000)
    authors: tuple[str, ...] = Field(default=(), max_length=200)
    journal_title: str | None = Field(default=None, max_length=500)
    publication_date: date | None = None
    publication_date_source_text: str | None = Field(default=None, max_length=100)
    work_type: LiteratureWorkType = LiteratureWorkType.UNVERIFIED
    peer_review_status: PeerReviewStatus = PeerReviewStatus.UNVERIFIED
    review_status_source_id: str | None = None
    identifiers: tuple[LiteratureIdentifier, ...] = Field(default=(), max_length=20)
    provenance: LiteratureProvenance
    abstract: str | None = Field(default=None, max_length=50000)
    content_level: ContentLevel = ContentLevel.METADATA_ONLY
    access_status: AccessStatus = AccessStatus.RESTRICTED_OR_UNKNOWN
    full_text_available: bool = False
    full_text_urls: tuple[str, ...] = Field(default=(), max_length=20)
    license_expression: str | None = Field(default=None, max_length=500)
    license_status: LicenseStatus = LicenseStatus.NOT_REPORTED
    linked_publication_versions: tuple[LinkedPublicationVersion, ...] = Field(default=(), max_length=20)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def scientific_status_and_access_are_explicit(self):
        if self.peer_review_status is PeerReviewStatus.PEER_REVIEWED and not self.review_status_source_id:
            raise ValueError("peer-reviewed status requires an explicit verification source.")
        if (
            self.peer_review_status is PeerReviewStatus.PREPRINT
            and self.work_type is not LiteratureWorkType.PREPRINT
        ):
            raise ValueError("preprint review status requires the preprint work type.")
        if self.abstract is not None and self.content_level is ContentLevel.METADATA_ONLY:
            raise ValueError("records containing an abstract must declare abstract-only or full-text content level.")
        if self.content_level is ContentLevel.FULL_TEXT and not self.full_text_available:
            raise ValueError("full-text evidence requires explicit full-text availability.")
        if (self.license_expression is None) != (self.license_status is LicenseStatus.NOT_REPORTED):
            raise ValueError("license status must distinguish reported terms from absent license metadata.")
        if any(not url.startswith("https://") for url in self.full_text_urls):
            raise ValueError("literature links must use HTTPS.")
        return self

    @property
    def prominent_warning(self) -> str | None:
        return PREPRINT_WARNING if self.peer_review_status is PeerReviewStatus.PREPRINT else None


class LiteratureSearchDisclosure(OperationModel):
    provider_id: str
    provider_name: str
    destination: str
    query: str = Field(min_length=1, max_length=500)
    transmitted_categories: tuple[str, ...] = ("researcher-entered literature query",)
    project_data_included: bool = False
    notice: str = (
        "The query will be sent to the named external literature service. NeuroScope will not attach recordings, "
        "project metadata, unpublished notes, or signal samples. Confirm that the query itself contains no confidential "
        "or unpublished research before continuing."
    )


class LiteratureConsentDecision(OperationModel):
    approved: bool
    exact_query_reviewed: bool
    no_project_data_or_recordings: bool
    no_unpublished_or_confidential_research: bool

    @property
    def authorizes_search(self) -> bool:
        return all((
            self.approved,
            self.exact_query_reviewed,
            self.no_project_data_or_recordings,
            self.no_unpublished_or_confidential_research,
        ))


class LiteratureSearchAuthorization(OperationModel):
    authorization_id: str = Field(min_length=1, max_length=100)
    provider_id: str
    query_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    authorized_by: str = Field(min_length=1, max_length=120)
    authorized_at_utc: datetime
    consent_receipt_id: str = Field(min_length=1, max_length=100)
    authorization_signature: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("authorized_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("authorized_at_utc must be timezone-aware and expressed in UTC.")
        return value


class LiteratureSearchResponse(OperationModel):
    provider_id: str
    query_sha256: str
    retrieved_at_utc: datetime
    records: tuple[LiteratureRecord, ...] = Field(max_length=25)


class ManualSearchLink(OperationModel):
    provider_id: Literal["google_scholar_manual"] = "google_scholar_manual"
    query: str = Field(min_length=1, max_length=500)
    url: str = Field(pattern=r"^https://scholar\.google\.com/scholar\?q=")
    mode: Literal["user_initiated_browser_search"] = "user_initiated_browser_search"
    scraping_backend: Literal[False] = False
    notice: str = "Opening this link sends the displayed query to Google Scholar in the user's browser; NeuroScope does not scrape results."


class LiteratureProvider(Protocol):
    provider_id: str
    provider_name: str
    destination: str

    def search(
        self,
        query: str,
        *,
        authorization: LiteratureSearchAuthorization,
        limit: int,
    ) -> LiteratureSearchResponse: ...


class LiteratureServiceError(ValueError):
    """Literature request could not safely be completed."""


class LiteratureConsentDenied(LiteratureServiceError):
    pass


class LiteratureSearchService:
    """Application gate that obtains fresh, query-bound consent before provider access."""

    def __init__(self, providers: tuple[LiteratureProvider, ...]) -> None:
        self._providers = {provider.provider_id: provider for provider in providers}
        if len(self._providers) != len(providers):
            raise ValueError("literature provider IDs must be unique.")

    def search(
        self,
        provider_id: str,
        query: str,
        *,
        consent_prompt: Callable[[LiteratureSearchDisclosure], LiteratureConsentDecision],
        researcher_id: str,
        limit: int = 10,
    ) -> LiteratureSearchResponse:
        provider = self._provider(provider_id)
        normalized_query = _validate_query(query)
        if not 1 <= limit <= 25:
            raise ValueError("literature result limit must be between 1 and 25.")
        disclosure = LiteratureSearchDisclosure(
            provider_id=provider.provider_id,
            provider_name=provider.provider_name,
            destination=provider.destination,
            query=normalized_query,
        )
        decision = consent_prompt(disclosure)
        if not isinstance(decision, LiteratureConsentDecision) or not decision.authorizes_search:
            raise LiteratureConsentDenied("researcher declined external literature search; no request was sent.")
        authorization = _authorization(provider.provider_id, normalized_query, researcher_id)
        return provider.search(normalized_query, authorization=authorization, limit=limit)

    def _provider(self, provider_id: str) -> LiteratureProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise LiteratureServiceError(f"literature provider {provider_id!r} is unavailable.") from exc


Transport = Callable[[str, float], Mapping[str, Any]]


class EuropePMCProvider:
    """Small metadata/abstract-only adapter for the official Europe PMC REST search API."""

    provider_id = EUROPE_PMC_ID
    provider_name = "Europe PMC"
    destination = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    api_version = "rest-search-v1"

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        timeout_s: float = 10.0,
        minimum_interval_s: float = 1.0,
    ) -> None:
        if not math.isfinite(timeout_s) or not math.isfinite(minimum_interval_s) or timeout_s <= 0 or minimum_interval_s < 0:
            raise ValueError("timeout must be positive and minimum request interval nonnegative.")
        self._transport = transport or _http_json_get
        self.timeout_s = timeout_s
        self.minimum_interval_s = minimum_interval_s
        self._last_request_at = 0.0

    def search(
        self,
        query: str,
        *,
        authorization: LiteratureSearchAuthorization,
        limit: int,
    ) -> LiteratureSearchResponse:
        normalized_query = _validate_query(query)
        _validate_authorization(authorization, self.provider_id, normalized_query)
        if not 1 <= limit <= 25:
            raise ValueError("literature result limit must be between 1 and 25.")
        self._wait_for_rate_limit()
        params = urllib.parse.urlencode(
            {"query": normalized_query, "format": "json", "pageSize": str(limit), "resultType": "core"}
        )
        url = f"{self.destination}?{params}"
        try:
            payload = self._transport(url, self.timeout_s)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise LiteratureServiceError("Europe PMC request failed or timed out; no result was returned.") from exc
        result_list = payload.get("resultList")
        if not isinstance(result_list, Mapping) or not isinstance(result_list.get("result", []), list):
            raise LiteratureServiceError("Europe PMC returned an invalid result structure.")
        retrieved_at = datetime.now(timezone.utc)
        query_hash = _query_hash(normalized_query)
        records = tuple(
            _europe_pmc_record(item, query_hash=query_hash, retrieved_at=retrieved_at)
            for item in result_list.get("result", [])[:limit]
            if isinstance(item, Mapping)
        )
        return LiteratureSearchResponse(
            provider_id=self.provider_id,
            query_sha256=query_hash,
            retrieved_at_utc=retrieved_at,
            records=records,
        )

    def _wait_for_rate_limit(self) -> None:
        remaining = self.minimum_interval_s - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class CrossrefDOIVerifier:
    """Read-only DOI lookup; Crossref metadata does not itself prove peer review."""

    provider_id = CROSSREF_ID
    provider_name = "Crossref"
    destination = "https://api.crossref.org/works"
    api_version = "crossref-works-v1"

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        timeout_s: float = 10.0,
        minimum_interval_s: float = 1.0,
    ) -> None:
        if not math.isfinite(timeout_s) or not math.isfinite(minimum_interval_s) or timeout_s <= 0 or minimum_interval_s < 0:
            raise ValueError("timeout must be positive and minimum request interval nonnegative.")
        self._transport = transport or _http_json_get
        self.timeout_s = timeout_s
        self.minimum_interval_s = minimum_interval_s
        self._last_request_at = 0.0

    def verify_doi(
        self,
        doi: str,
        *,
        authorization: LiteratureSearchAuthorization,
    ) -> LiteratureRecord:
        normalized_doi = _normalize_doi(doi)
        query = f"doi:{normalized_doi}"
        _validate_authorization(authorization, self.provider_id, query)
        self._wait_for_rate_limit()
        url = f"{self.destination}/{urllib.parse.quote(normalized_doi, safe='') }"
        try:
            payload = self._transport(url, self.timeout_s)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise LiteratureServiceError("Crossref DOI lookup failed or timed out.") from exc
        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise LiteratureServiceError("Crossref returned an invalid DOI record.")
        returned_doi = _normalize_doi(str(message.get("DOI", "")))
        if returned_doi != normalized_doi:
            raise LiteratureServiceError("Crossref did not verify the requested DOI.")
        retrieved_at = datetime.now(timezone.utc)
        title_values = message.get("title", [])
        title = next((item for item in title_values if isinstance(item, str)), None) if isinstance(title_values, list) else None
        authors = tuple(
            " ".join(part for part in (str(item.get("given", "")).strip(), str(item.get("family", "")).strip()) if part)
            for item in message.get("author", [])
            if isinstance(item, Mapping)
        ) if isinstance(message.get("author", []), list) else ()
        published = message.get("published-online") or message.get("published-print") or message.get("created")
        pubdate = _crossref_date(published)
        identifiers = (LiteratureIdentifier(
            scheme="doi",
            value=returned_doi,
            verification=IdentifierVerification.VERIFIED_BY_CROSSREF,
            verification_source_id=CROSSREF_ID,
            verified_at_utc=retrieved_at,
        ),)
        links = tuple(
            LinkedPublicationVersion(
                identifier=LiteratureIdentifier(scheme="doi", value=_relation_doi(relation), verification=IdentifierVerification.SOURCE_ASSERTED),
                relation=relation_name,
                source_id=CROSSREF_ID,
                source_url=f"https://api.crossref.org/works/{urllib.parse.quote(_relation_doi(relation), safe='')}",
            )
            for relation_name, relations in (message.get("relation", {}) or {}).items()
            if isinstance(relations, list)
            for relation in relations
            if isinstance(relation, Mapping) and _relation_doi(relation)
        )
        query_hash = _query_hash(query)
        return LiteratureRecord(
            record_id=returned_doi,
            title=title,
            authors=authors,
            journal_title=_first_string(message.get("container-title")),
            publication_date=pubdate,
            publication_date_source_text=_crossref_date_text(published),
            work_type=_crossref_work_type(message.get("type")),
            peer_review_status=PeerReviewStatus.UNVERIFIED,
            identifiers=identifiers,
            provenance=LiteratureProvenance(
                provider_id=self.provider_id,
                provider_version=self.api_version,
                source_record_id=returned_doi,
                source_url=str(message.get("URL", f"https://doi.org/{returned_doi}")),
                retrieved_at_utc=retrieved_at,
                query_sha256=query_hash,
            ),
            content_level=ContentLevel.METADATA_ONLY,
            access_status=AccessStatus.RESTRICTED_OR_UNKNOWN,
            full_text_available=False,
            license_expression=_crossref_license(message.get("license")),
            license_status=(LicenseStatus.REPORTED_BY_SOURCE if _crossref_license(message.get("license")) else LicenseStatus.NOT_REPORTED),
            linked_publication_versions=links,
            warnings=("Crossref metadata verifies DOI registration, not peer-review status or scientific validity.",),
        )

    def _wait_for_rate_limit(self) -> None:
        remaining = self.minimum_interval_s - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class AuthorizedLiteratureClient:
    """Consent gate for searches and DOI verification across approved providers."""

    def __init__(self, search_providers: tuple[LiteratureProvider, ...], doi_verifier: CrossrefDOIVerifier | None = None):
        self.search_service = LiteratureSearchService(search_providers)
        self.doi_verifier = doi_verifier

    def search(self, provider_id: str, query: str, *, consent_prompt, researcher_id: str, limit: int = 10):
        return self.search_service.search(
            provider_id,
            query,
            consent_prompt=consent_prompt,
            researcher_id=researcher_id,
            limit=limit,
        )

    def verify_doi(self, doi: str, *, consent_prompt, researcher_id: str) -> LiteratureRecord:
        if self.doi_verifier is None:
            raise LiteratureServiceError("Crossref DOI verification is not configured.")
        normalized = _normalize_doi(doi)
        query = f"doi:{normalized}"
        disclosure = LiteratureSearchDisclosure(
            provider_id=CROSSREF_ID,
            provider_name=self.doi_verifier.provider_name,
            destination=self.doi_verifier.destination,
            query=query,
        )
        decision = consent_prompt(disclosure)
        if not isinstance(decision, LiteratureConsentDecision) or not decision.authorizes_search:
            raise LiteratureConsentDenied("researcher declined external DOI lookup; no request was sent.")
        authorization = _authorization(CROSSREF_ID, query, researcher_id)
        return self.doi_verifier.verify_doi(normalized, authorization=authorization)


def _europe_pmc_record(item: Mapping[str, Any], *, query_hash: str, retrieved_at: datetime) -> LiteratureRecord:
    raw_record_id = item.get("id") or item.get("pmid") or item.get("doi")
    if not isinstance(raw_record_id, (str, int)) or not str(raw_record_id).strip():
        raise LiteratureServiceError("Europe PMC result has no stable source record identifier.")
    source_record_id = str(raw_record_id).strip()
    record_url = f"https://europepmc.org/article/{urllib.parse.quote(str(item.get('source', 'MED')), safe='')}/{urllib.parse.quote(source_record_id, safe='')}"
    is_preprint = item.get("isPreprint") is True or str(item.get("isPreprint", "")).lower() == "true"
    doi = item.get("doi")
    pmid = item.get("pmid")
    pmcid = item.get("pmcid")
    identifiers: list[LiteratureIdentifier] = []
    for scheme, value in (("doi", doi), ("pmid", pmid), ("pmcid", pmcid)):
        if isinstance(value, str) and value.strip():
            identifiers.append(LiteratureIdentifier(scheme=scheme, value=value.strip(), verification=IdentifierVerification.SOURCE_ASSERTED))
    abstract = item.get("abstractText")
    if not isinstance(abstract, str) or not abstract.strip():
        abstract = None
    raw_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", []) if isinstance(item.get("fullTextUrlList"), Mapping) else []
    fulltext_urls = tuple(
        str(link.get("url")) for link in raw_urls
        if isinstance(link, Mapping) and isinstance(link.get("url"), str) and link["url"].startswith("https://")
    ) if isinstance(raw_urls, list) else ()
    open_access = item.get("isOpenAccess") is True or str(item.get("isOpenAccess", "")).lower() == "true"
    if open_access and fulltext_urls:
        access_status = AccessStatus.OPEN_ACCESS_FULL_TEXT_AVAILABLE
    elif abstract is not None:
        access_status = AccessStatus.ABSTRACT_ONLY
    else:
        access_status = AccessStatus.RESTRICTED_OR_UNKNOWN
    publication_date = _parse_date(item.get("firstPublicationDate"))
    work_type = LiteratureWorkType.PREPRINT if is_preprint else _epmc_work_type(item.get("pubTypeList"))
    review_status = PeerReviewStatus.PREPRINT if is_preprint else PeerReviewStatus.UNVERIFIED
    pub_versions = _explicit_related_versions(item.get("relatedRecords"), record_url)
    warnings = (
        (PREPRINT_WARNING,) if is_preprint else ()
    ) + (("Europe PMC's record does not independently establish peer-review status.",) if not is_preprint else ())
    return LiteratureRecord(
        record_id=source_record_id,
        title=item.get("title") if isinstance(item.get("title"), str) else None,
        authors=_split_authors(item.get("authorString")),
        journal_title=item.get("journalTitle") if isinstance(item.get("journalTitle"), str) else None,
        publication_date=publication_date,
        publication_date_source_text=item.get("firstPublicationDate") if isinstance(item.get("firstPublicationDate"), str) else None,
        work_type=work_type,
        peer_review_status=review_status,
        review_status_source_id=EUROPE_PMC_ID if is_preprint else None,
        identifiers=tuple(identifiers),
        provenance=LiteratureProvenance(
            provider_id=EUROPE_PMC_ID,
            provider_version=EuropePMCProvider.api_version,
            source_record_id=source_record_id,
            source_url=record_url,
            retrieved_at_utc=retrieved_at,
            query_sha256=query_hash,
        ),
        abstract=abstract,
        content_level=ContentLevel.ABSTRACT_ONLY if abstract is not None else ContentLevel.METADATA_ONLY,
        access_status=access_status,
        full_text_available=bool(open_access and fulltext_urls),
        full_text_urls=fulltext_urls,
        license_expression=(item.get("license").strip() if isinstance(item.get("license"), str) and item.get("license").strip() else None),
        license_status=(LicenseStatus.REPORTED_BY_SOURCE if isinstance(item.get("license"), str) and item.get("license").strip() else LicenseStatus.NOT_REPORTED),
        linked_publication_versions=pub_versions,
        warnings=warnings,
    )


def _explicit_related_versions(raw: Any, source_url: str) -> tuple[LinkedPublicationVersion, ...]:
    """Use only explicit relation entries supplied in the provider payload."""
    if not isinstance(raw, list):
        return ()
    result = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        value = item.get("doi") or item.get("id")
        relation = item.get("type") or item.get("relation")
        if isinstance(value, str) and value.strip() and isinstance(relation, str) and relation.strip():
            result.append(LinkedPublicationVersion(
                identifier=LiteratureIdentifier(scheme="doi", value=_normalize_doi(value), verification=IdentifierVerification.SOURCE_ASSERTED),
                relation=relation,
                source_id=EUROPE_PMC_ID,
                source_url=source_url,
            ))
    return tuple(result)


def _authorization(provider_id: str, query: str, researcher_id: str) -> LiteratureSearchAuthorization:
    name = researcher_id.strip()
    if not name:
        raise ValueError("researcher identity is required for external literature consent.")
    data = {
        "authorization_id": str(uuid4()),
        "provider_id": provider_id,
        "query_sha256": _query_hash(query),
        "authorized_by": name,
        "authorized_at_utc": datetime.now(timezone.utc),
        "consent_receipt_id": str(uuid4()),
    }
    signature = hmac.new(
        _AUTHORIZATION_SECRET,
        "|".join(str(data[key]) for key in ("authorization_id", "provider_id", "query_sha256", "authorized_by", "authorized_at_utc", "consent_receipt_id")).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    with _AUTHORIZATION_LOCK:
        _ISSUED_AUTHORIZATIONS.add(data["authorization_id"])
    return LiteratureSearchAuthorization(
        **data,
        authorization_signature=signature,
    )


def _validate_authorization(auth: LiteratureSearchAuthorization, provider_id: str, query: str) -> None:
    if auth.provider_id != provider_id or auth.query_sha256 != _query_hash(query):
        raise LiteratureConsentDenied("literature authorization does not match this provider and exact query.")
    data = {
        "authorization_id": auth.authorization_id,
        "provider_id": auth.provider_id,
        "query_sha256": auth.query_sha256,
        "authorized_by": auth.authorized_by,
        "authorized_at_utc": auth.authorized_at_utc,
        "consent_receipt_id": auth.consent_receipt_id,
    }
    expected = hmac.new(
        _AUTHORIZATION_SECRET,
        "|".join(str(data[key]) for key in ("authorization_id", "provider_id", "query_sha256", "authorized_by", "authorized_at_utc", "consent_receipt_id")).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(auth.authorization_signature, expected):
        raise LiteratureConsentDenied("literature authorization was not issued by the consent gate.")
    with _AUTHORIZATION_LOCK:
        if auth.authorization_id not in _ISSUED_AUTHORIZATIONS:
            raise LiteratureConsentDenied("literature authorization is unavailable or has already been used.")
        _ISSUED_AUTHORIZATIONS.remove(auth.authorization_id)


def _validate_query(query: str) -> str:
    normalized = " ".join(query.split()) if isinstance(query, str) else ""
    if not normalized or len(normalized) > 500:
        raise ValueError("literature query must contain 1 to 500 non-whitespace characters.")
    if re.search(r"(?:[A-Za-z]:\\|\\\\[^\\]+\\|/Users/|/home/)", normalized):
        raise ValueError("local filesystem paths are not permitted in a literature query.")
    return normalized


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _normalize_doi(value: str) -> str:
    normalized = urllib.parse.unquote(value.strip())
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    if not re.fullmatch(r"10\.\d{4,9}/\S+", normalized, flags=re.IGNORECASE):
        raise ValueError("a DOI in the form 10.xxxx/suffix is required.")
    return normalized.lower()


def _http_json_get(url: str, timeout_s: float) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "NeuroScope-AI/0.1 (read-only scientific literature metadata; no project data)",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LiteratureServiceError(f"literature service returned HTTP {exc.code}.") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LiteratureServiceError("literature service returned malformed JSON.") from exc
    if not isinstance(payload, Mapping):
        raise LiteratureServiceError("literature service returned a non-object response.")
    return payload


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _crossref_date(value: Any) -> date | None:
    if not isinstance(value, Mapping):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
        return None
    if len(parts[0]) != 3:
        return None
    try:
        return date(*[int(item) for item in parts[0]])
    except (ValueError, TypeError):
        return None


def _crossref_date_text(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
        return None
    try:
        return "-".join(f"{int(item):02d}" if index else f"{int(item):04d}" for index, item in enumerate(parts[0]))
    except (ValueError, TypeError):
        return None


def _first_string(value: Any) -> str | None:
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, str)), None)
    return value if isinstance(value, str) else None


def _split_authors(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    return tuple(part.strip() for part in re.split(r"\s*;\s*", value) if part.strip())


def _epmc_work_type(value: Any) -> LiteratureWorkType:
    types = {str(item).lower() for item in value} if isinstance(value, list) else set()
    if "journal article" in types:
        return LiteratureWorkType.JOURNAL_ARTICLE
    if types:
        return LiteratureWorkType.OTHER
    return LiteratureWorkType.UNVERIFIED


def _crossref_work_type(value: Any) -> LiteratureWorkType:
    if value == "journal-article":
        return LiteratureWorkType.JOURNAL_ARTICLE
    if isinstance(value, str) and "preprint" in value.lower():
        return LiteratureWorkType.PREPRINT
    return LiteratureWorkType.OTHER if isinstance(value, str) and value else LiteratureWorkType.UNVERIFIED


def _relation_doi(relation: Mapping[str, Any]) -> str:
    identifier = relation.get("id")
    if not isinstance(identifier, str):
        return ""
    try:
        return _normalize_doi(identifier)
    except ValueError:
        return ""


def manual_google_scholar_link(query: str) -> ManualSearchLink:
    """Return a user-opened search URL; this function does not contact Google Scholar."""
    normalized = _validate_query(query)
    url = "https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": normalized})
    return ManualSearchLink(query=normalized, url=url)


def _crossref_license(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    entries = []
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("URL"), str) and item["URL"].strip():
            entries.append(item["URL"].strip())
    return "; ".join(dict.fromkeys(entries)) or None
