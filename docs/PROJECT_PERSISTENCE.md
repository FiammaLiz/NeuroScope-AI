# Scientific project persistence and recovery

## Project contents

Each project folder is local and self-contained. `project.json` stores a UUID project identity and manifest schema version. `dataset_profile.json` stores the selected dataset's shape, dtype, orientation and metadata confidence. `dataset_source_identity.json` stores a source descriptor, including format, selected MATLAB dataset, structure, file size, modification time and a SHA-256 digest of bounded first/middle/last file blocks. The digest is **partial**; it is not a whole-file content proof. `execution_history.json` stores serialized operation inputs, parameters, metadata snapshots, outputs, warnings, approval provenance and execution IDs. Scientific arrays and interval tables remain in project-owned artifact directories, outside JSON execution records.

Dataset identity, project identity and a live `DataReference` are different things. Dataset IDs in historical records identify the source profile used at execution time. A `DataReference` also contains a session UUID and expires when the source is unloaded or the application closes. Reopening a project never revives an old handle or approval token.

## Open and recovery behavior

Use **Open project** to select a folder. NeuroScope validates the manifest, profile, history structure and artifact directories before changing the active project. Unsupported manifest versions or unreadable top-level files leave the current workspace untouched. A successful open restores compatible typed execution records and explorer entries but does not rerun analyses. Records whose operation schema or implementation version is unavailable remain in the stored history and are reported as unsupported. Saved outputs can be inspected while their raw source is offline; source-dependent navigation stays disabled.

Interrupted artifact staging directories are reported as incomplete and are never automatically removed. Missing, corrupt, unsupported-schema, or lineage-mismatched artifacts are reported instead of regenerated. Artifact validation currently checks manifest identity, expected file entries, payload availability, array shape/dtype, and lineage against execution history. The current artifact formats do not contain payload checksums; accidental bit corruption that preserves a valid NPY/JSON structure may not be detected.

## Reconnect a source dataset

Load the candidate recording after opening a project. NeuroScope compares the selected format, internal dataset path, shape, dtype, file size and bounded partial content fingerprint. A candidate with matching checks must still be explicitly confirmed because unexamined bytes could differ. A legacy project without a fingerprint also requires explicit researcher confirmation and records that uncertainty. A mismatch can only be loaded as a **separate dataset**; it receives a distinct dataset ID and does not rebind historical analyses. Reconnection preserves the saved metadata values and their confirmed/inferred/unknown states, then registers the source with a fresh session handle. No sampling rate, units or signal type are inferred during recovery.

Paths are retained locally to make reconnecting practical. Assistant summaries omit absolute paths, and raw samples, project files and artifacts are not sent to model providers. Assistant conversation history and transient UI selection are not project scientific provenance and are not currently persisted. Project execution history is available to the assistant's read-only tools after opening a project, with historical/source/artifact status stated explicitly.

## Legacy compatibility and backups

Projects created before `project.json` are opened as legacy projects without adding a manifest during open. Their execution history and structurally valid artifacts can be recovered; source identity is unknown until the researcher confirms a candidate. Old operation snapshots that do not match the currently registered operation and implementation versions are preserved but not loaded into active typed explorers.

Back up the complete project folder while NeuroScope is closed. This includes the manifest, profile and source descriptor, execution history, and every artifact subdirectory. Do not back up only JSON history: scientific arrays and interval tables are stored separately. Keep an independent copy of source recordings if reproducibility requires access to the original data.

## Desktop walkthrough

1. Create a project and load a dataset.
2. Run only reviewed and approved analyses; execution records and artifacts are stored in the selected project.
3. Close NeuroScope after active executions and artifact reads finish.
4. Start NeuroScope and select **Open project**.
5. Inspect the recovered detector, waveform, ISI and MUAtrace entries. They are historical results; opening them does not execute an operation.
6. Load the original dataset when sample navigation is needed, review the source comparison, and explicitly confirm a reconnect. Old session references remain invalid; the current application uses the new registration.

## Known limitations

- The partial source fingerprint is bounded and can miss modifications outside sampled blocks; it is used only to inform an explicit confirmation.
- Artifact payload checksums and automatic repair are not implemented.
- Project metadata is stored as a current dataset profile rather than a versioned decision log; earlier metadata revisions are represented only in execution snapshots that captured them.
- UI visibility/layout preferences and chat conversations are intentionally transient.
- Legacy operation implementations may no longer be available; those records remain inspectable as JSON summaries but cannot be used as current typed operation inputs.
