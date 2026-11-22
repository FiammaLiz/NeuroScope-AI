# NeuroScope AI Roadmap

## Intended MVP target

This is the product target, not a statement that all listed capabilities are implemented.

One complete interactive, offline-capable synthetic neural-signal workflow: dataset loading/characterization; independent raw exploration/navigation; comparison of two distinct spike detectors with selectable positive/negative polarity; waveform extraction and ISI inspection; PSTH as the single activity analysis; minimal persistence/provenance sufficient to save and reopen the synthetic experiment; and a basic approval gate introduced with the minimal operation contract before any analysis operation can execute. Preprocessing is optional and is not a prerequisite for detector comparison; include it in the MVP only if a specific scientific justification is agreed. Users adjust parameters, compare configurations and inspect intermediate results, with immediate feedback where feasible. The MVP excludes MUA, complete project memory, reusable pipeline graphs, broad assistant capabilities, multimodal processing, advanced population methods, automatic module generation, local LLM support, literature search and autoencoders. Phases 4 and 5 complete and generalize the MVP foundations.

This original MVP boundary predates later implementation work: MUA, project persistence/pipelines, assistant providers, literature libraries, and Knowledge Packs have since been added. PSTH and the other deferred analyses below remain unimplemented. Phase descriptions retain their original planned sequence and acceptance criteria; the implementation status labels and snapshot below supersede future-tense wording where work has since landed.

## Phases

### Phase 0 — Specification (implemented; specifications remain maintained)

Define product principles, modalities, data contracts, assistant approval model and validation strategy. Deliver documentation, not scientific features.

### Phase 1 — Ingestion and characterization (implemented and test-covered)

Stabilize NPY/CSV/TXT/MAT parsing, variable selection, validation, orientation and metadata states. Keep complex/nested MATLAB structures unsupported until defined and tested. Exit on focused synthetic reader tests and clear limitations.

### Phase 2 — Exploration and preprocessing (exploration implemented; preprocessing remains optional/planned)

Add independent dataset inspection and navigation: metadata/status display, channel selection, timestamp-aware views where rate is confirmed, interval selection and raw signal visualization. This phase must work without a preprocessing filter. Preprocessing remains optional; if justified, specify its scientific purpose and settings separately, preserve filter phase/artifact masks/provenance, and validate it without making it a prerequisite for detector comparison.

### Phase 3 — First analyses (analysis workflow implemented; PSTH remains planned)

Define the minimal operation contract and basic approval gate before exposing analysis execution. Every analysis operation, whether initiated directly or proposed by an assistant, must pass through the gate and require explicit user confirmation before execution. Then build the deterministic synthetic workflow; compare the current median-centered-MAD detector with the owner-authored median-absolute-amplitude detector and allow positive/negative polarity; add waveform extraction, ISI summaries and PSTH. Preprocessing is not required to validate detector comparison. Acceptance: (1) report sensitivity, false positives, timing error and waveform recovery separately against ground truth, with numeric targets agreed before validation; (2) fixed seed/configuration reproduces generated data and results; (3) invalid parameters and missing metadata are rejected; (4) interactive changes correctly update dependent outputs and intermediate results are inspectable; (5) PSTH event/trial schema, units, bins, empty-trial handling, edges and normalization are defined and tested; (6) project save/reopen preserves the generated recording, generator configuration/seed, ground-truth tables and analysis results as distinct artifacts; regeneration from seed must not silently replace the stored recording; (7) the gate blocks every operation until explicit confirmation. Phase 4 expands approval/provenance and persistence into full project memory and reusable graphs; Phase 5 adds typed assistant proposals, deterministic contract validation, richer inspection/recommendations and memory integration. Numeric detection targets remain unresolved and must be agreed before validation. Synthetic acceptance is not biological validation.

### Phase 4 — Project persistence and pipelines (implemented; structured decision memory remains incomplete)

Expand MVP persistence into complete structured project memory, reusable operation graphs and pipelines, scientific decision records (choice, rationale, relevant rejected alternatives and unresolved questions), full provenance, dependency tracking/invalidation, robust project reopening and reproducibility. Keep decisions distinct from chat. Avoid overwrite-by-default and validate deterministic replay.

### Phase 5 — AI assistant (provider and proposal workflows implemented; live reliability unverified)

Expand the MVP approval gate into provider-independent orchestration, typed analysis proposals, deterministic validation against operation contracts, richer dataset inspection/recommendations, and structured project-memory integration. Validate parameter/metadata checks and approval enforcement. Preserve offline use and consent-based external context.

### Phase 6 — Multimodal analysis (planned)

Add audio, pressure and behavioral adapters, clock synchronization, acoustic/spectral features, event-related coupling and correlation. Test known offsets/drift and lag direction synthetically. Keep correlation descriptive unless inference is calibrated.

### Phase 7 — Advanced methods (planned)

Evaluate population analyses, PCA/autoencoders, DTW/similarity, statistical workflows in Python and R, adaptive MUA, local models, generated modules, plugins and alternate UIs. Treat R as a first-class statistical environment. Use language-independent contracts where practical, with Python/R adapters, dataset/design metadata, dependency/version checks, reproducible execution, provenance, consistent results and visualization. Do not automatically translate legacy scripts. Any future method reuse requires source-level authorship, dependency, license, permission, and scientific-assumption review. Do not publish unpublished experimental results or confidential laboratory methodology without explicit researcher approval.

## Initial synthetic demo configuration (proposal, not platform defaults)

Use a small configuration record with every value explicit and editable: four channels, two seconds per recording, 30,000 samples/second, amplitude unit `µV`, configurable per-channel Gaussian noise standard deviation, and configurable spike templates with polarity, peak amplitude, component widths and sample/time placement. Add optional event markers and trial/condition tables. These values describe a first demo profile only; they are not inferred metadata or universal defaults.

Keep four artifacts distinct: (1) generated recording samples, saved as the experiment data; (2) generator configuration and seed; (3) ground-truth spike/event tables; and (4) analysis results. Seed-based regeneration is an explicit reproducibility operation and never implicitly replaces the stored recording.

## Recommended first implementation task

Implement and test the deterministic synthetic recording generator. It creates the controlled data and ground truth required by detector comparison, waveform extraction, ISI and PSTH validation, without altering the existing detector. The task should return the recording, configuration/seed and ground-truth tables as separate values; tests should verify deterministic regeneration, positive/negative events, sample/time mapping, parameter validation and passage through the ordinary save/load path. Saving or regenerating must not conflate or silently replace the stored recording.

## Cross-cutting operation contract and responsiveness

Each operation declares inputs/required metadata; parameters/validation; outputs/output metadata; warnings/limitations; provenance; and dependencies on prior operations. The basic approval gate is implemented, and assistant proposals use that same execution path. No analysis runs without explicit approval. Pipelines are composable and reusable across interfaces; methods are implemented once. Support interactive parameter exploration and intermediate inspection. For large recordings consider segmented visualization/lazy loading; profile before choosing optimizations and reuse unaffected intermediates when valid.

## Scientific provenance and reuse boundaries

The calcMUA method provenance and the distinction between its method, MATLAB implementation, and NeuroScope Python implementation are documented in `SCIENTIFIC_SCOPE.md` and `README.md`. The original signed-peak Python reference was authored by Fiamma Liz Leites; NeuroScope maintains a separate implementation based on that reference. These attributions do not establish redistribution rights for any reference file or imply endorsement. Review each source's authorship, license, dependencies, permissions, and scientific assumptions before reuse. Do not publish confidential laboratory protocols or unpublished results without explicit approval.

## Dependencies and milestones

Metadata/timebases precede preprocessing and event analyses. Synchronization and trial schemas precede PSTH/coupling. Ground truth precedes detector performance claims. Provenance precedes caching/assistant memory. Registry validation precedes tool execution. Statistical hierarchy and upstream selection review precede inferential reporting.

## Current implementation status (November 2026)

### Implemented and automated-test covered

The current repository contains a PySide6 desktop application; numeric NPY/CSV/TXT/classic MAT and lazy MATLAB v7.3/HDF5 loading; dataset exploration; typed metadata and scientific operation contracts; explicit prepare/review/approve/execute lifecycle; two separate spike detector operations; synthetic recordings and polarity-aware evaluation; waveform extraction; ISI analysis; MATLAB-derived `calcMUA`; MUAtrace averaging; persistent projects/artifacts; reusable pipelines; provider-neutral assistant tools and operation proposals; OpenAI and Ollama adapters; local scientific libraries; and Knowledge Packs. The tests use deterministic fixtures and fake provider transports where relevant. They verify software contracts, not biological validity.

### Implemented but not independently or scientifically validated

The v7.3 tests use synthetic HDF5 fixtures; genuine MATLAB-generated files have not been validated. `calcMUA` has no MATLAB reference-output parity result. Mocked provider tests do not establish live OpenAI or Ollama response reliability. No biological performance claim or real-recording validation is made here.

### Planned or deferred

PSTH, broad preprocessing, multimodal synchronization, audio/behavior analyses, advanced population and statistical methods, OCR, embedding/vector retrieval, broad literature synthesis, and generated analysis modules remain planned or deferred. `PRODUCT_VISION.md` and `SCIENTIFIC_SCOPE.md` describe these boundaries and the scientific questions that remain open.

## Risks and open decisions

Risks include silent metadata ambiguity, an over-large MVP, mistaking synthetic validation for biological validation, collaborator/third-party code provenance, uncalibrated surrogate inference, dependent windows, UI performance and privacy. Address them with explicit metadata, one synthetic end-to-end workflow, independent validated implementations, calibrated nulls, and local-by-default processing.

Decide: synthetic signal units and initial demo; portable project format; scale/performance target; next file formats; AI provider and consent UX; file-specific source ownership, licensing, and redistribution questions. Do not include materials whose distribution rights remain unresolved.
