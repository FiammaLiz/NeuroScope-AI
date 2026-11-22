# NeuroScope AI Product Vision

NeuroScope AI is an extensible desktop environment for exploring biological signals and conducting reproducible analyses. It combines a local scientific engine with an optional scientific assistant that can inspect data, explain methods, and prepare analyses for researcher approval.

## Audience and workflows

The product serves experimental researchers, students, academic laboratories, computational scientists, and developers of signal-processing methods. It should support local data import, metadata characterization, signal exploration, explicit preprocessing and analysis, result comparison, provenance review, and project reopening. It is not restricted to birdsong or electrophysiology and does not replace scientific judgment or make association equivalent to causation.

## Principles

- Keep confirmed, inferred, and unknown metadata distinct.
- Keep ordinary workflows offline-capable and raw data local by default.
- Record inputs, metadata, preprocessing, parameters, versions, seeds, outputs, and warnings.
- Require researcher approval before assistant-proposed analyses or pipeline changes execute.
- Support interactive parameter adjustment, immediate feedback where computationally feasible, configuration comparison, and inspection of intermediate results.
- Preserve scientific decisions and rationale separately from conversational history.
- Keep algorithms independent of the GUI and model provider.
- Support modality-specific metadata and clocks rather than flattening them into one assumed schema.
- Treat research scripts as reference material, not automatically production-ready features.

## Proposed architecture

Separate adapters, dataset profiles, preprocessing, composable operations, a validated registry, pipeline orchestration, results/visualization, provenance, assistant services, and the PySide6 UI. Every operation declares inputs/required metadata, parameters/validation, outputs/output metadata, warnings/limitations, provenance, and dependencies on prior operations. Pipelines are reusable across interfaces; algorithms are not reimplemented for each UI. Support parameter adjustment, fast feedback where feasible, configuration comparison, intermediate inspection and assistant discussion. For large recordings, consider segmented views and lazy loading; profile before optimizing, and reuse unaffected intermediates when valid.

## Implementation snapshot (November 2026)

The repository contains a PySide6 desktop application, scientific operation contracts and approval runner, project persistence and reusable pipelines, two separate spike detectors, waveform extraction, ISI analysis, MATLAB-derived `calcMUA`, compatible MUAtrace averaging, synthetic data/evaluation, assistant providers/tools, literature libraries, and Knowledge Packs. The source tree and automated tests cover these capabilities, but tests do not establish biological validity or independent scientific correctness.

| Status | Current evidence and boundary |
|---|---|
| Implemented and test-covered | Numeric ingestion and exploration; approval-bound scientific operations; synthetic generator/evaluation; the analyses listed above; project history/artifacts/pipelines; assistant tool and proposal boundaries; local literature libraries and Knowledge Packs. |
| Implemented, not independently validated | MATLAB v7.3 lazy loading has synthetic HDF5 fixture coverage, but no genuine MATLAB-generated recording was validated. `calcMUA` has no MATLAB numerical-parity result. Provider adapters have deterministic mocked coverage; this is not evidence of live-model reliability. |
| Planned | PSTH; multimodal synchronization and analysis; advanced population methods; OCR, embedding/vector retrieval and broader literature synthesis; generated analysis modules. |

The MVP paragraph below records the intended product target, not a claim that every item is currently available. In particular, PSTH remains planned. Check `SCIENTIFIC_SCOPE.md` and `ROADMAP.md` for the same status distinctions and validation limits.

### Intended MVP target

One complete interactive, offline-capable synthetic neural-signal workflow: loading and characterization, independent raw exploration, comparison of distinct spike detectors, waveform/ISI inspection, one activity analysis, minimal persistence/provenance, and explicit approval before analysis execution. Preprocessing is optional and detector validation does not depend on a filter unless a specific scientific justification is agreed. This target describes product scope, not current completion. Multimodal processing, advanced population methods, cross-correlation, and automatic module generation remain later or deferred work.

**Decided:** PySide6 remains the current UI; AI is optional; researchers approve execution; computations should work offline; the product is modality-extensible.

**Open:** portable project format, supported recording standards, plugin trust model, cache policy, interaction latency target, and additional provider support.
