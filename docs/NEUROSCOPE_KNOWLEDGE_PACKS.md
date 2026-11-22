# NeuroScope Knowledge Packs

## Purpose and trust boundary

A Knowledge Pack is a portable, versioned collection of selected bibliographic references, authorized document files, and researcher-authored methodological resources. Packs can provide locally retrieved context to the existing Scientific Knowledge Engine without changing model weights, adding assistant tools, or changing scientific operation permissions. Every imported pack and its contents remain untrusted evidence; they cannot execute code, install packages, alter prompts or approve operations.

Pack identity, authorship, version and content fingerprint are declarations from the pack manifest. The SHA-256 fingerprint detects accidental alteration against the manifest but is not a digital signature and does not authenticate the author or publisher. A pack is not a certificate of scientific quality.

## Format and versioning

The portable `.nskpack` file is a ZIP container with `manifest.json` and only explicitly referenced `payloads/<resource-id>/<filename>` entries. The manifest is validated against schema version 1, semantic pack version, minimum/maximum NeuroScope version, known resource kinds, identifiers, dependencies, metadata, privacy classification, redistribution declaration and payload fingerprints. Duplicate archive names, paths, links, encrypted entries, path traversal, unreferenced entries, unsupported schema/index versions, oversized entries/manifests/packages and excessive compression ratios are rejected. Payloads are inspected in memory and installed under generated library paths; archive paths are never extracted as filesystem paths.

Resource selection is explicit. The API can package references, document references, methodology, terminology, protocols, interpretation warnings, researcher notes, benchmark material, profile references and optional pack dependencies. The GUI currently lets researchers create a pack from selected library documents and optionally add one researcher-authored note. Pack versions are immutable within a local catalog: reusing an ID/version with different scientific contents is rejected. A different contents set requires a new version. Importing an already installed ID/version is rejected rather than replaced.

## Export and import

Export defaults to **reference-only**. It contains metadata, identifiers, source provenance and explicitly authored note text, but no imported document payload. Content-inclusive export is a separate mode that requires confirmation of redistribution authorization. If any selected document is private or has unknown access, a second confirmation is required. License metadata remains unknown when unknown; confirmations do not fabricate a license. Open access does not automatically mean redistribution is permitted.

Import first previews the manifest, resource list, privacy, licensing and warnings. The researcher chooses a local organizational scope. Installation creates a separate library and does not activate it or mark its documents assistant-visible. The researcher must select the pack library in the active corpus and explicitly enable each document for assistant visibility. This reuses the existing retrieval and external document-consent boundary. Pack resources and scientific metadata are immutable after installation; local visibility settings may change, and a content update requires a new pack version. Remove affects only the selected pack library; deactivate by unchecking it in the active corpus. No linked documents are downloaded.

Manifest imports preserve IDs and fingerprints but have no searchable payload. Payload bundles are independently validated before passages are indexed. Original resource IDs, source library/document IDs, pack ID/version, package hash, and source provenance remain in local records and retrieved passage provenance.

## Privacy and scope

Pack privacy is conservatively labeled from selected document metadata unless explicitly declared more restrictive. An unknown status remains unknown. Scope labels (core, personal, laboratory, project) and pack privacy are local organization/visibility labels only. They do not implement authentication, encryption, multi-user access control, network sharing, synchronization, or centralized storage. Project-specific libraries are stored in that project. Opening another local catalog is an explicit researcher action.

External providers receive no pack file, source file, local path or raw signal. A retrieved passage from an external pack library can be sent only after the existing fresh, per-request consent names that document. Consent is not inferred from pack import, activation, visibility, or an open-access link. The local assistant remains functional offline.

## Retrieval and interpretation

Imported content uses the existing deterministic local lexical index. It does not add an agent tool or bypass assistant allowlisting. Exact passages include library/document/passage IDs and pack ID/version in provenance. A preprint retains the warning **“Preprint — not peer reviewed.”** Metadata-only references are not searchable as text. Pack-authored instructions remain untrusted document content. Keep general model knowledge, profile guidance, retrieved evidence, researcher interpretation and project results distinct.

## Limitations

The archive fingerprint is integrity checking, not a signature. There is no marketplace, account system, collaborative authentication, network synchronization, pack signing, linked-document download, OCR, full-text RAG embedding, or automatic dependency installation. PDF extraction requires the optional `pypdf` extra and may lose layout, equations, tables and captions. The environment used for the implementation did not include `pypdf`; PDF package extraction has not been validated against an actual PDF here. No claim is made that packs improve model accuracy or that included methods are biologically validated.
