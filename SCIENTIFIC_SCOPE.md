# NeuroScope AI Scientific Scope

This document describes scientific scope and distinguishes implemented software from planned research capabilities. NeuroScope AI is a research prototype, not a validated biological or clinical analysis suite.

## Modalities and requirements

| Modality | Typical representation | Essential metadata |
|---|---|---|
| Extracellular voltage | Continuous samples × channels; event/waveform tables | Rate, gain/units, channel map, reference and filter history |
| Spike trains | Event times with unit/channel/trial identifiers | Timebase, unit provenance, event and trial definitions |
| LFP, EEG, ECG, EMG | Continuous time series × channels | Rate, units, labels, reference and acquisition history |
| Audio | Waveform samples × audio channels | Rate, scaling, channel interpretation, clock/synchronization |
| Pressure/other sensors | Continuous samples or timestamped measurements | Units, calibration, rate/timestamps, sensor identity |
| Events/behavior | Timestamps, annotations, trajectories, custom variables | Clock, semantics, units, provenance and hierarchy |

Do not assume modalities share sample rates, units, clock origins, or event definitions. Preserve modality-specific structure.

## Analysis catalog

### Exploration and preprocessing

Inspect dimensions, channels, timestamps, events, gaps, and uncertain metadata; visualize continuous segments and compare signals. Candidate operations include filtering, normalization, baseline correction, artifact marking, resampling, segmentation, and raw/processed comparison. Preserve operation order and parameters; distinguish causal and zero-phase filtering. Never silently replace or discard artifacts. Cutoffs must respect rate and Nyquist limits.

### Spike detection and quality

Potential methods include robust noise estimation, selectable polarity, thresholds, peak/prominence/refractory constraints, waveform extraction, interactive visualization, sensitivity analysis, and manual review. Keep estimators and conventions separately named and parameterized. Reference Python `spike_detection.py` estimates noise as median absolute amplitude / 0.6745, supports signed thresholds, prominence and an amplitude cap. The repository detector uses median-centered MAD and negative local minima. These estimators are not equivalent: state centering, polarity, signal units, threshold definition, local-extremum rule, prominence, refractory and amplitude constraints independently. Do not flatten multichannel inputs silently. Interactive controls should update results/plots promptly where feasible, compare configurations and expose intermediate noise, threshold and detection results. Waveform/ISI/refractory metrics are quality evidence, not definitive false-positive measures. Detection is not sorting.

### MUA and firing activity

Keep separate: thresholded continuous-voltage MUA, rates from detected timestamps, Gaussian-kernel spike activity, and fixed/adaptive estimates. Distinguish pooled spike-weighted from trial-balanced estimates. Record units, normalization, bandwidth, threshold, time grid and trial policy. Adaptive methods in the reference archive are experimental, not universal defaults.

### Trial/event analyses

Raster, PSTH, rates, event windows and kernel density require event/trial identity, alignment convention, synchronization and analysis window. Preserve empty trials and specify bin edges and denominators.

### Audio/acoustic features

Waveforms, spectrograms, Mel features, envelopes, onset/offset, segmentation and cross-modal alignment are candidate methods. Do not hardcode universal rate/window/hop values; document frame timing.

### Spectral, correlation and synchrony

PSD, spectrogram, band power and phase methods depend on signal type, sampling rate and context. LFP requires suitable continuous data. Keep autocorrelation, continuous cross-correlation, spike-train correlograms, firing-rate correlations, sliding windows and lag estimation distinct. Specify lag sign, time conversion, masks and alignment; correlation is not causation. Additional methods remain deferred pending scientific validation and provenance, authorship, and licensing review. Unpublished experimental results and confidential method details are not documented here. Do not reuse third-party or collaborator-authored code without appropriate permission.

### Population, similarity and statistics

Future methods include PCA, autoencoders, latent trajectories, Euclidean/MSE/SSIM/Fréchet metrics, DTW, density/mixture models and mixed-effects models. Record scaling, training/test splits and leakage risks. Report DTW warping rather than hiding timing differences. Respect experimental hierarchy and avoid pseudoreplication. Autoencoder training details and statistical model appropriateness are not established by utility filenames alone.

## Implemented state, unverified behavior, and planned scope

### Scientific-method provenance

The calcMUA scientific method was developed by Dr. Cecilia Herbert. Fiamma Liz Leites independently wrote the MATLAB implementation `calcmua_verFi.m` as a methodological implementation of Herbert's algorithm. NeuroScope's Python implementation is separate and based on Leites' MATLAB implementation. It is not presented as Herbert-authored code or as endorsed by Herbert. The method is used in the following collaborative preprint by Herbert and Leites, which is scientific context for calcMUA, not publication of Herbert's original source code or a claim that the paper first introduced the method: Leites, F. L., Herbert, C. T., Boari, S., Cignoli, F. I., Mindlin, G. B., & Amador, A. (2026). *Low-dimensional neural dynamics underlying rhythmic vocal behavior in songbirds*. Preprint. [https://doi.org/10.64898/2026.03.28.715006](https://doi.org/10.64898/2026.03.28.715006). Numerical parity with MATLAB-generated reference outputs is unverified.

The original Python `spike_detection.py` reference implementation was authored by Fiamma Liz Leites; the NeuroScope signed-peak operation is a separate implementation based on it. Source-method attribution does not assert that the reference code or method has independent biological validation.

### Implemented and covered by automated tests

The current source includes NPY/CSV/TXT/classic MAT and MATLAB v7.3/HDF5 numeric loading, dataset exploration, explicit metadata confidence states, two distinct spike detector operations, synthetic ground-truth evaluation, waveform extraction, ISI analysis, the MATLAB-derived `calcMUA` operation, MUAtrace averaging, operation approval/provenance, project persistence, reusable pipelines, assistant providers/tools/proposals, local literature libraries, and Knowledge Packs. The GUI exposes the implemented workflows. Tests establish software behavior for their fixtures and contracts; they are not biological validation.

### Implemented but not independently or scientifically validated

MATLAB v7.3 behavior is tested using synthetic HDF5 fixtures; no genuine MATLAB-generated recording has been validated. Python `calcMUA` has no MATLAB reference-output parity result. OpenAI/Ollama orchestration has deterministic mock coverage; no claim of live-model reliability follows. Literature extraction and retrieval have automated tests, but search relevance is not scientific evidence quality. Real recordings, laboratory workflows, and human-reviewed scientific performance remain unvalidated.

### Planned or deferred

PSTH, broad preprocessing, multimodal clock synchronization, audio/behavior analysis, advanced population/statistical analyses, OCR, embedding/vector retrieval, broad full-text literature synthesis, and generated analysis modules are not implemented as general capabilities. Candidate methods elsewhere in this document describe research scope, not available operations.

Every composable operation should declare inputs and required metadata; parameters and validation; outputs and output metadata; warnings and limitations; provenance; and dependencies on prior operations. Pipelines should be reusable across interfaces without duplicating algorithms. Intermediate outputs should be inspectable. Reuse unaffected upstream results when valid, and consider segmented visualization/lazy loading for large recordings; profile before selecting optimizations.

The original Phase 3 acceptance criteria called for a PSTH activity analysis, detector metrics against synthetic ground truth, deterministic results, parameter/metadata validation, interactive updates, save/reopen, and explicit approval. Detector operations, synthetic evaluation, interactive analysis, persistence, and approval are now implemented and covered by automated tests. PSTH and agreed numeric performance targets remain outstanding. Phase 4 persistence/pipelines and Phase 5 assistant proposal foundations are present; complete structured scientific decision history remains incomplete. These status notes describe software delivery only. Synthetic success is not biological validation.

## Synthetic validation

Start with a configurable deterministic multichannel noisy-voltage recording, known spike times/amplitudes/polarities, and optional events/trials/conditions. Store the generated recording, generator configuration/seed, ground-truth tables and analysis results as distinct artifacts. Regeneration from a seed supports reproducibility but must never implicitly replace the original stored recording. Sampling rate, units, waveform shape/amplitude and noise are per-generator configuration values, not platform defaults. Dataset exploration is independent of preprocessing; do not require a filter for detector comparison absent a specific scientific justification. Measure sensitivity, false positives, timing error, waveform recovery, threshold/noise sensitivity, PSTH alignment and seeded reproducibility. Synthetic results do not establish biological validity. Later use public datasets and private data locally with permissions and review.

## Open scientific questions

Confirm amplitude calibration/filter provenance; canonical event/trial and clock-drift schema; detector and baseline convention per modality; exact adaptive MUA definitions; surrogate null/error control; experimental hierarchy; and dataset scale/performance targets.
