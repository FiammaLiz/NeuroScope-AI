"""Optional assistant orchestration, isolated from scientific operation modules."""

from .contracts import (
    AssistantProposal,
    AssistantToolResult,
    AssistantTurn,
    ChatMessage,
)
from .knowledge_packs import (
    KnowledgePackDependency,
    KnowledgePackError,
    KnowledgePackManifest,
    KnowledgePackPreview,
    KnowledgePackResource,
    KnowledgePackService,
)
from .library import (
    LibraryDocument,
    LibraryDocumentMetadata,
    LibraryManifest,
    LibraryPassage,
    LiteratureLibraryStore,
    RetrievedLibraryPassage,
    ScientificLibrary,
)
from .literature import (
    AuthorizedLiteratureClient,
    CrossrefDOIVerifier,
    EuropePMCProvider,
)
from .orchestration import AssistantOrchestrator
from .profiles import ScientificProfile, ScientificProfileRegistry, builtin_profiles
from .providers import OpenAIProvider
from .scientific_benchmark import (
    benchmark_suite,
    run_profile_comparison,
    score_manual_assessment,
)
from .tools import AssistantApplicationState, AssistantToolService

__all__ = [
    "AssistantApplicationState",
    "AssistantOrchestrator",
    "AssistantProposal",
    "AssistantToolResult",
    "AssistantToolService",
    "AssistantTurn",
    "AuthorizedLiteratureClient",
    "ChatMessage",
    "CrossrefDOIVerifier",
    "EuropePMCProvider",
    "KnowledgePackDependency",
    "KnowledgePackError",
    "KnowledgePackManifest",
    "KnowledgePackPreview",
    "KnowledgePackResource",
    "KnowledgePackService",
    "LibraryDocument",
    "LibraryDocumentMetadata",
    "LibraryManifest",
    "LibraryPassage",
    "LiteratureLibraryStore",
    "OpenAIProvider",
    "RetrievedLibraryPassage",
    "ScientificLibrary",
    "ScientificProfile",
    "ScientificProfileRegistry",
    "benchmark_suite",
    "builtin_profiles",
    "run_profile_comparison",
    "score_manual_assessment",
]
