# Local Scientific Literature Library

## Scope and trust boundaries

The Literature workspace stores researcher-selected literature locally and provides deterministic lexical retrieval. It does not fine-tune a model, make scientific decisions, or add scientific execution authority. Local document instructions are untrusted content. Search scores mean lexical relevance only. A retrieved statement is evidence to inspect, not proof that it is correct.

Libraries are separate IDs with one of four organizational scopes: core, personal, laboratory, or project. The active project catalog is stored under that project. Researchers may explicitly open a separate local catalog folder to reuse libraries across projects; opening another project resets the selection to that project's catalog. These scopes are local organization and visibility boundaries only. They do not provide authentication, encryption, multi-user access control, network sharing, or synchronization. Do not open a private catalog in a project unless its use there is authorized.

The model sees only documents in libraries selected by the researcher and documents explicitly checked as assistant-visible. A model query cannot add libraries or documents to that selection. Imported document names, DOI, authors, dates, publication/review state, access state, license, source provenance, content fingerprint, extraction state, index version, and passage ranges are kept in local records. Local source paths are not passed to providers.

## Import and storage

Use the **Literature** workspace to create a named library, choose its scope, select multiple authorized PDF/TXT/Markdown files, inspect extraction status, mark individual documents assistant-visible, remove a selected document, search, view exact passages, open a retained source, and export or import a metadata manifest. TXT and Markdown use bounded UTF-8 decoding. PDF text uses the optional `pypdf` extra; scanned-image PDFs are unsupported because OCR is not provided. Extraction is bounded to 40 MiB per source and 2 million extracted characters. PDF equations, tables, captions, layout and reading order can be degraded and are explicitly warned about.

The source-copy question is explicit on import. Choosing not to retain a source stores extracted passage text and metadata only. Choosing to retain it makes a project-local copy. Removal asks before deleting that copy. Replacing an existing document is an explicit API action using its stable document ID and reindexes that source; a content fingerprint change is never silently treated as unchanged. On reopening, indexes are validated and reused without extraction. Changed retained sources are marked `changed_needs_review`; corrupted indexes produce an error and are not reconstructed automatically.

`pypdf` is an optional dependency because TXT/Markdown libraries work without it:

```powershell
# Run from the repository root.
.venv\Scripts\python.exe -m pip install -e ".[literature]"
```

No package is installed by the application. The developer environment used for implementation did not contain `pypdf`; PDF extraction tests therefore cover its clear unavailable-dependency error, not a real PDF parse. Install the optional extra locally to enable PDF parsing.

## Retrieval and evidence

`LiteratureLibraryStore` is the local catalog interface. Its typed library, document, passage and retrieval models carry stable identifiers and provenance. Search is deterministic BM25 over the explicitly selected corpus, bounded to 50,000 passages, 500 query characters, five assistant passages, 1,400 characters per passage, and three seconds for assistant-tool retrieval. The GUI's local search can return up to ten passages. Filters can constrain libraries, document IDs, source types and publication status. Passage IDs are content/location-derived within a document and index version. The index is JSON and is never sent as a file.

PDF passages retain page numbers. TXT/Markdown passages retain half-open character ranges in normalized extracted text; Markdown headings are recorded as sections. The stored exact passage text is shown in the GUI with document ID, passage ID, page/range, DOI if provided, publication status, access status, and retrieval score. Abstract-only evidence is labeled. Preprints display **“Preprint — not peer reviewed.”** DOI presence does not establish review. A linked journal DOI remains a relationship only and is not assumed scientifically equivalent.

Task 20 `LiteratureRecord` objects may be explicitly imported as metadata-only entries or with an explicitly selected abstract. The source URL, provider/version, retrieval timestamp, identifier verification, publication state, and license/access metadata are preserved. This import method performs no online request. A metadata-only record has no searchable passage.

## Assistant and privacy

The only assistant literature capability is the allowlisted read-only `search_local_scientific_library` tool, with typed `query` and bounded `limit`. It searches only researcher-selected active libraries and assistant-visible documents. It cannot open paths, change corpus selection, execute operations, or modify documents. The immutable assistant prompt says retrieved evidence, general knowledge, interpretation, hypothesis, and uncertainty must remain distinct; citations must use returned document/passage IDs. Tool results remain untrusted data.

Loopback Ollama retrieval and inference stay local. Before OpenAI or a non-loopback Ollama request, the GUI asks for a fresh per-request decision for each named assistant-visible document that may provide passages. A second provider-context consent remains in place. Permission is in-memory for that request and does not carry over when the provider changes. Unchecked documents are excluded from external tool results. No raw recordings, full PDFs, local paths, or source files are sent by this tool. This permission is not a claim about provider retention; use the provider's current account data controls.

Access status is public, private, or unknown only when explicitly supplied; the default for local file import is unknown. The library interface allows the researcher to set access and review-status labels with an attributed local decision record; manually supplied peer-review status is not independently verified by this local UI. A public status is not a substitute for checking copyright, license, or transmission permissions. Researchers are responsible for importing only authorized material. No automatic Europe PMC search, full-text download, Google Scholar scraping, OCR, cloud library storage, or collaborative authentication is implemented.

## Manifest portability

**Export metadata manifest** writes JSON metadata, fingerprints, schema/index versions, extraction status and provenance, but no source document, passage text, or local absolute path. Import validates the manifest and creates metadata-only `manifest_only` records that are not searchable until the researcher imports authorized source documents. IDs are preserved, and collisions are rejected instead of replacing an existing library. A separate `export_payload_bundle(..., researcher_confirmed=True)` API exists for retained source files; it refuses overwrite and requires an explicit confirmation argument. Payload export does not run automatically or occur during manifest export.

Libraries and indexes are local project data; do not treat scope names as security controls. Project-specific libraries are isolated by project root. A separately opened catalog is an explicit researcher action and must be reopened deliberately in each project.

Versioned `.nskpack` NeuroScope Knowledge Packs can package selected library references and researcher-authored methodology. Reference-only export is the default; content-inclusive export requires explicit redistribution authorization and an additional confirmation when selected sources are private or access-unknown. Imported packs are separate inactive libraries, and their documents remain assistant-hidden until individually enabled. See [NEUROSCOPE_KNOWLEDGE_PACKS.md](NEUROSCOPE_KNOWLEDGE_PACKS.md) for archive validation, immutable versions, licensing, privacy, compatibility, and integrity limitations.

## Deterministic evaluation

`examples/literature_evaluation_cases.json` contains synthetic policy/retrieval cases including neuroscience interpretation examples. It contains no real citations and does not assess model reliability. Automated tests verify retrieval, source and passage provenance, unknown/preprint/abstract labels, cross-library selection, visibility filtering, document replacement, metadata manifest safety, payload-export confirmation, reopen without reindex, changed-source detection, corrupted-index handling, privacy filtering, offline behavior, and GUI access. Model-generated answer quality requires a separate manually reviewed evaluation; these tests make no accuracy-improvement claim.
