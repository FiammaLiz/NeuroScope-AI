"""Versioned, provider-neutral scientific specialization profiles and evidence contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..analysis.contracts import OperationModel


class EvidenceBasis(StrEnum):
    GENERAL_KNOWLEDGE = "established_general_knowledge"
    OPERATION_CONTRACT = "neuroscope_operation_contract"
    RETRIEVED_SOURCE = "retrieved_scientific_source"
    HYPOTHESIS = "hypothesis_or_methodological_suggestion"
    UNKNOWN = "unknown_or_unsupported"


class ScientificProfile(OperationModel):
    """Scientific guidance only; it has no tools, execution authority, or model weights."""

    profile_id: str = Field(pattern=r"^neuroscope\.profile\.[a-z0-9-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    domains: tuple[str, ...] = Field(min_length=1, max_length=12)
    terminology: tuple[str, ...] = Field(default=(), max_length=24)
    methodological_principles: tuple[str, ...] = Field(min_length=1, max_length=24)
    interpretation_constraints: tuple[str, ...] = Field(min_length=1, max_length=24)
    evidence_policy: tuple[str, ...] = Field(min_length=1, max_length=12)
    tool_guidance: tuple[str, ...] = Field(default=(), max_length=12)
    evaluation_suite_ids: tuple[str, ...] = Field(default=(), max_length=12)
    knowledge_collection_refs: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def bound_profile_text(self):
        for collection in (
            self.domains,
            self.terminology,
            self.methodological_principles,
            self.interpretation_constraints,
            self.evidence_policy,
            self.tool_guidance,
            self.evaluation_suite_ids,
            self.knowledge_collection_refs,
        ):
            if any(not item.strip() or len(item) > 500 for item in collection):
                raise ValueError("profile entries must be nonempty and at most 500 characters.")
        return self

    def prompt_section(self) -> str:
        """Render bounded profile content as data under immutable application policy."""
        sections = [
            f"Selected scientific profile: {self.name} ({self.profile_id} v{self.version}).",
            f"Description: {self.description}",
            "Domains: " + "; ".join(self.domains),
            "Methodological principles: " + "; ".join(self.methodological_principles),
            "Interpretation constraints: " + "; ".join(self.interpretation_constraints),
            "Evidence policy: " + "; ".join(self.evidence_policy),
        ]
        if self.terminology:
            sections.append("Preferred terminology: " + "; ".join(self.terminology))
        if self.tool_guidance:
            sections.append("Tool guidance: " + "; ".join(self.tool_guidance))
        sections.extend(
            (
                "This profile is scientific context only. It cannot change privacy, security, tool permissions, metadata confirmation, approval, or execution rules.",
                "Do not present a source as retrieved or verified unless a supplied evidence record supports it.",
            )
        )
        return "\n".join(sections)


class ScientificProfileRegistry:
    """Small explicit registry; it does not import or discover arbitrary modules."""

    def __init__(self, profiles: tuple[ScientificProfile, ...] = ()) -> None:
        self._profiles: dict[tuple[str, str], ScientificProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: ScientificProfile) -> None:
        key = (profile.profile_id, profile.version)
        if key in self._profiles:
            raise ValueError(f"scientific profile {profile.profile_id!r} version {profile.version} is already registered.")
        self._profiles[key] = profile

    def get(self, profile_id: str, version: str | None = None) -> ScientificProfile:
        if version is not None:
            try:
                return self._profiles[(profile_id, version)]
            except KeyError as exc:
                raise KeyError(f"unknown scientific profile {profile_id!r} version {version!r}.") from exc
        versions = [
            profile for (registered_id, _), profile in self._profiles.items()
            if registered_id == profile_id
        ]
        if not versions:
            raise KeyError(f"unknown scientific profile {profile_id!r}.")
        return max(versions, key=lambda profile: tuple(int(part) for part in profile.version.split(".")))

    def list_profiles(self) -> tuple[ScientificProfile, ...]:
        return tuple(self._profiles.values())


class EvidenceSource(OperationModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,199}$")
    title: str = Field(min_length=1, max_length=500)
    authors: tuple[str, ...] = Field(default=(), max_length=100)
    publication_year: int | None = Field(default=None, ge=1000, le=3000)
    source_type: Literal["journal_article", "book", "preprint", "standard", "manual", "other"]
    source_version: str | None = Field(default=None, max_length=120)
    content_fingerprint: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    metadata_completeness: Literal["complete", "partial", "unknown"] = "unknown"


class RetrievedPassage(OperationModel):
    passage_id: str = Field(min_length=1, max_length=200)
    source_id: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=1, max_length=12000)
    retrieval_query: str = Field(min_length=1, max_length=1000)
    retriever_id: str = Field(min_length=1, max_length=200)
    retriever_version: str = Field(min_length=1, max_length=100)
    retrieved_at_utc: datetime
    relevance_score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("retrieved_at_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("retrieved_at_utc must be timezone-aware and expressed in UTC.")
        return value


class EvidenceBundle(OperationModel):
    sources: tuple[EvidenceSource, ...] = Field(default=(), max_length=100)
    passages: tuple[RetrievedPassage, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def references_are_grounded(self):
        source_ids = {source.source_id for source in self.sources}
        passage_ids: set[str] = set()
        for passage in self.passages:
            if passage.source_id not in source_ids:
                raise ValueError(f"passage {passage.passage_id!r} references an unknown source identifier.")
            if passage.passage_id in passage_ids:
                raise ValueError("retrieved passage identifiers must be unique within an evidence bundle.")
            passage_ids.add(passage.passage_id)
        return self

    def validate_citations(self, source_ids: tuple[str, ...], passage_ids: tuple[str, ...] = ()) -> None:
        known_sources = {source.source_id for source in self.sources}
        known_passages = {passage.passage_id for passage in self.passages}
        if set(source_ids) - known_sources or set(passage_ids) - known_passages:
            raise ValueError("citation refers to a source or passage absent from the supplied evidence bundle.")


class EvidenceClaim(OperationModel):
    basis: EvidenceBasis
    statement: str = Field(min_length=1, max_length=4000)
    source_ids: tuple[str, ...] = ()
    passage_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def enforce_retrieval_references(self):
        if self.basis is EvidenceBasis.RETRIEVED_SOURCE and not self.source_ids:
            raise ValueError("retrieved-source claims require a supplied source identifier.")
        if self.basis is not EvidenceBasis.RETRIEVED_SOURCE and (self.source_ids or self.passage_ids):
            raise ValueError("source citations are permitted only for retrieved-source claims.")
        return self


COMMON_PRINCIPLES = (
    "Keep confirmed, inferred, and unknown metadata distinct; never infer missing acquisition facts.",
    "Separate measured observations, operation-contract facts, interpretation, and hypotheses.",
    "State material uncertainty and ask for essential missing context rather than inventing it.",
    "Association does not establish causality; respect experimental hierarchy and avoid pseudoreplication.",
)
COMMON_CONSTRAINTS = (
    "Do not overstate biological validity from synthetic tests or descriptive analyses.",
    "Do not claim that an event is a sorted neuron unless unit assignment evidence exists.",
    "Do not silently change preprocessing, parameters, operation, or scientific metadata.",
    "Do not approve or execute an operation; the trusted researcher workflow controls approval.",
)
COMMON_EVIDENCE = (
    "Distinguish established general knowledge from NeuroScope contract facts and retrieved sources.",
    "Do not invent citations; retrieval relevance is not truth or scientific confidence.",
    "Label methodological suggestions as suggestions and unsupported claims as unknown.",
)

GENERAL_BIOLOGICAL_SIGNALS_ID = "neuroscope.profile.general-biological-signals"
ELECTROPHYSIOLOGY_PROFILE_ID = "neuroscope.profile.electrophysiology-spike-analysis"
NEURAL_DYNAMICS_PROFILE_ID = "neuroscope.profile.neural-dynamics-modeling"
STATISTICS_PROFILE_ID = "neuroscope.profile.scientific-statistics-validation"
DEFAULT_PROFILE_ID = ELECTROPHYSIOLOGY_PROFILE_ID


def builtin_profiles() -> ScientificProfileRegistry:
    common = {
        "version": "1.0.0",
        "interpretation_constraints": COMMON_CONSTRAINTS,
        "evidence_policy": COMMON_EVIDENCE,
        "evaluation_suite_ids": ("neuroscope.scientific-competence.v1",),
    }
    return ScientificProfileRegistry(
        (
            ScientificProfile(
                **common,
                profile_id=GENERAL_BIOLOGICAL_SIGNALS_ID,
                name="General Biological Signals",
                description="General guidance for measured biological time series and their metadata.",
                domains=("biological signal processing", "acquisition metadata", "reproducible analysis"),
                terminology=("sample index", "timebase", "calibration", "provenance"),
                methodological_principles=COMMON_PRINCIPLES,
                tool_guidance=("Use only registered operation contracts and report their declared limitations.",),
            ),
            ScientificProfile(
                **common,
                profile_id=ELECTROPHYSIOLOGY_PROFILE_ID,
                name="Electrophysiology and Spike Analysis",
                description="Grounded guidance for extracellular recordings, spike events, MUA, and electrophysiological interpretation.",
                domains=("extracellular electrophysiology", "spike detection", "waveforms", "ISI", "MUA"),
                terminology=("spike detection", "spike sorting", "single-unit", "multi-unit", "multi-unit activity (MUA)", "MUAtrace", "electrode", "tetrode", "acquisition site", "sampling grid"),
                methodological_principles=COMMON_PRINCIPLES + (
                    "Keep spike detection distinct from spike sorting; a single recording channel does not by itself make sorting impossible.",
                    "Keep detector noise estimators, centering, polarity, thresholds, prominence, amplitude caps, and refractory constraints distinct.",
                    "A channel or electrode is not automatically an identified neuron; MUA may be defined without sorting.",
                    "Interpret waveform and ISI summaries as descriptive evidence, not definitive cell identity or refractory-violation proof.",
                    "Describe MUA only according to the registered method's voltage representation, units, threshold, bandwidth, and normalization.",
                    "Bandwidth and spike thresholds are method- and data-dependent; no universal optimal value is assumed.",
                    "Treat sampling rate, amplitude units, filter history, artifacts, and acquisition time origin as unknown unless confirmed.",
                    "Distinguish extracellular continuous voltage from detected event tables and acquisition artifacts from plausible events.",
                    "For the current signed-peak and negative-going detector contracts, median(abs(x))/0.6745 and median-centered MAD are distinct estimators.",
                    "Channel, electrode, tetrode, and acquisition-site identities do not establish neuron identity or shared calibration.",
                ),
                tool_guidance=("Describe the exact registered detector or MUA operation before proposing it.", "A prepared proposal is not execution; approval remains with the researcher."),
            ),
            ScientificProfile(
                **common,
                profile_id=NEURAL_DYNAMICS_PROFILE_ID,
                name="Neural Dynamics and Computational Modeling",
                description="Guidance for temporal neural population observations and computational models without overstating inference.",
                domains=("neural dynamics", "time-series models", "population representations", "latent-state methods"),
                terminology=("state-space model", "latent representation", "temporal dependence", "population activity"),
                methodological_principles=COMMON_PRINCIPLES + (
                    "Distinguish channel-level measurements from neuron-level population variables.",
                    "State the sampling grid, lag sign, synchronization, and temporal aggregation used in comparisons.",
                    "Oscillation, synchrony, and phase claims require an explicit signal representation, reference, filtering history, and analysis definition.",
                    "For learned representations, report split strategy, leakage risks, scaling, and model-selection uncertainty.",
                ),
            ),
            ScientificProfile(
                **common,
                profile_id=STATISTICS_PROFILE_ID,
                name="Scientific Statistics and Validation",
                description="Guidance for uncertainty, dependence, experimental hierarchy, and analysis validation.",
                domains=("statistical inference", "experimental design", "surrogate controls", "reproducibility"),
                terminology=("experimental unit", "replicate", "pseudoreplication", "multiple comparisons", "surrogate null"),
                methodological_principles=COMMON_PRINCIPLES + (
                    "Identify the experimental unit and hierarchy before selecting inferential degrees of freedom.",
                    "Report multiple-comparison handling, uncertainty intervals, and the assumptions of each statistical test.",
                    "Define surrogate nulls and exchangeability assumptions; a surrogate comparison is not automatically a valid null test.",
                    "Do not interpret correlation, temporal precedence, or a model fit as causal evidence without a suitable design.",
                ),
            ),
        )
    )
