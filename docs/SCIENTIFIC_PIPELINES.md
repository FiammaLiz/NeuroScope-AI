# Reproducible scientific pipelines

NeuroScope pipelines are versioned project records that compose registered scientific operations. They do not contain numerical implementations, NumPy arrays, live session handles, or approvals. Every new scientific node still uses the normal operation runner.

## Current operation compatibility

| Source | Destination | Contracted connection | Scientific meaning |
| --- | --- | --- | --- |
| `neuroscope.spike.detect-negative` | `neuroscope.spike.extract-waveforms` | `detection_events` → `detections` | Original detector events are adapted into the waveform event collection. |
| `neuroscope.spike.detect-signed-peaks` | `neuroscope.spike.extract-waveforms` | `detection_events` → `detections` | Same typed event handoff; detector identity is retained. |
| Either registered detector | `neuroscope.spike.analyze-isi` | `detection_events` → `events` | Event collection is adapted to the ISI input contract. |
| `neuroscope.mua.calc-mua` | `neuroscope.mua.average-traces` | `mua_trace` → `traces` | Previously computed compatible calcMUA artifacts are averaged. |

All signal operations take explicit dataset bindings. No other edges are inferred from operation names. In particular, calcMUA independently processes continuous signal and is not redefined as a detector-event transform; MUA averaging consumes calcMUA artifacts, never raw voltage.

## Definition and bindings

Schema version `1.0.0` stores a pipeline ID, revision, creation/update timestamps, descriptive name, ordered node records, registered operation IDs and versions, explicit parameter mappings, dependencies, and typed inputs. A dataset input stores a persistent `DatasetReference`, zero-based channel indices, and a half-open sample interval `[start, stop)`. It never stores the session handle from a live `DataReference`. At preparation time, the active session must contain exactly one registration matching the saved dataset identity; dimensions, channels, interval, metadata values and metadata confidence are checked again.

Node-output bindings name a specific upstream node and typed output port. The current ports are `detection_events` and `mua_trace`. Static artifact bindings are limited to explicit MUA artifact references. Reused execution nodes refer to an existing successful execution ID and preserve that record's original identity; they do not create a new approval or execution record.

The engine validates schema and operation versions, parameter models, required input fields, edge compatibility, missing dependencies, cycles, prerequisites, active dataset availability and metadata requirements. A graph can be structurally valid while nodes are blocked because a dataset is not active, metadata is unknown, or upstream results are not yet available. Topological ordering is deterministic, with node IDs breaking ties.

## Approval and execution

1. Create or open a pipeline definition in **Pipelines**. Add a registered operation with **Add Registered Operation**, or edit the JSON node list directly. Configure parameters explicitly; the operation picker does not insert parameter defaults.
2. Bind a selected channel and sample interval with **Bind Active Dataset**. Use **Connect Compatible Nodes** for one of the audited typed edges. **Select Saved MUA Artifact** inserts one lineage-verified MUA artifact; add at least two distinct compatible traces to an averaging node. **Reuse Saved Execution** adds a successful historical execution without rerunning it.
3. Validate the definition and review node readiness, warnings and blockers.
4. Save a new project-owned revision. Existing revisions are not overwritten.
5. Start a run. This records state only; it does not execute a node.
6. Select **Prepare Next Node** and review its exact registered operation/version, implementation version, typed inputs, parameters, metadata snapshot and warnings.
7. Select **Approve & Execute Prepared Node** and confirm the single-node approval dialog. Each later node requires a separate prepare and approval.
8. Inspect the execution result, provenance and node status before preparing a dependent node.

The assistant receives no pipeline approval or execution capability. Opening a saved definition or run snapshot does not execute, reapprove, reconnect a dataset, or regenerate missing artifacts. A prepared state is not approval; approvals remain in memory with the runner. A stale dataset handle or changed node definition blocks execution. If a node fails, its descendants are marked blocked; independent successful artifacts remain intact.

## Persistence and artifact reuse

Definitions are stored under `pipelines/<pipeline-id-digest>/revision-NNNNNN.json`; run-state snapshots are atomically updated under that pipeline's `runs` directory. Execution records remain in the existing execution history and artifacts remain in the existing artifact stores. These are separate records linked by execution and artifact IDs. Malformed and unsupported pipeline records are reported by project discovery and are not rewritten.

Reuse is explicit. A researcher can bind a compatible saved execution as a reused node or bind an explicit MUA artifact reference. Operation identity, semantic and implementation versions are checked; the target operation performs its normal artifact and scientific compatibility validation before execution. No automatic result cache substitutes for a requested operation.

## Supported example paths

For a synthetic detection workflow, first generate and save a recording with its original configuration, seed and ground truth retained separately. Bind the saved recording to a detector node, then connect that detector's `detection_events` output independently to waveform extraction and ISI analysis. Each node receives explicit parameters and individual approval. Ground truth is not passed to the detector.

Example node list (replace the dataset ID, interval and illustrative parameters with the actual project selection and scientific plan):

```json
{
  "pipeline_id": "example-synthetic-spike-workflow",
  "schema_version": "1.0.0",
  "name": "Synthetic detector, waveform and ISI workflow",
  "description": "Example only; parameter values are not NeuroScope defaults.",
  "revision": 1,
  "created_at_utc": "2026-01-01T00:00:00Z",
  "updated_at_utc": "2026-01-01T00:00:00Z",
  "nodes": [
    {
      "node_id": "detect",
      "operation_id": "neuroscope.spike.detect-negative",
      "operation_version": "1.0.0",
      "parameters": {"threshold_sd": -4.0},
      "depends_on": [],
      "reuse_execution_id": null,
      "inputs": [
        {
          "target_field": "signal",
          "source_kind": "dataset",
          "dataset": {"dataset_id": "replace-with-loaded-dataset-id"},
          "channels": [0],
          "sample_interval": {"start": 0, "stop": 10000}
        }
      ]
    },
    {
      "node_id": "waveforms",
      "operation_id": "neuroscope.spike.extract-waveforms",
      "operation_version": "1.0.0",
      "parameters": {"pre_samples": 12, "post_samples": 24},
      "depends_on": ["detect"],
      "reuse_execution_id": null,
      "inputs": [
        {
          "target_field": "signal",
          "source_kind": "dataset",
          "dataset": {"dataset_id": "replace-with-loaded-dataset-id"},
          "channels": [0],
          "sample_interval": {"start": 0, "stop": 10000}
        },
        {
          "target_field": "detections",
          "source_kind": "node_output",
          "source_node_id": "detect",
          "source_port": "detection_events"
        }
      ]
    },
    {
      "node_id": "isi",
      "operation_id": "neuroscope.spike.analyze-isi",
      "operation_version": "1.0.0",
      "parameters": {"polarity_mode": "separate"},
      "depends_on": ["detect"],
      "reuse_execution_id": null,
      "inputs": [
        {
          "target_field": "events",
          "source_kind": "node_output",
          "source_node_id": "detect",
          "source_port": "detection_events"
        }
      ]
    }
  ]
}
```

For the separate MUA workflow, create one `neuroscope.mua.calc-mua` node per selected channel with explicit `stdmin`, `mua_window_s`, `mode`, and `num_points`. Connect two or more compatible `mua_trace` outputs to `neuroscope.mua.average-traces`. Compatibility rules and the existing equal-channel arithmetic mean are unchanged.

For a separate MUA example, use two `neuroscope.mua.calc-mua` nodes bound to distinct explicitly selected channels, each with its own explicit `stdmin`, `mua_window_s`, `mode`, and `num_points`. Connect both with `mua_trace` ports to one `neuroscope.mua.average-traces` node. The values and selected channels must follow the recording and research protocol; no parameter values are universal defaults.

## Limitations

- The editor is JSON-based and supports the audited edges above; it is not a general plugin or arbitrary pipeline language.
- Every new node is executed sequentially; nodes are not parallelized.
- Run snapshots are inspectable after reopening, but in-memory approvals are never restored and a run is not automatically resumed.
- Dataset reconnection remains an explicit project action. Dataset IDs and optional fingerprints do not prove complete content identity.
- Historical result reuse requires the exact currently registered operation and implementation version.
- Synthetic validation does not establish biological validity.
- Pipeline metadata is not added to assistant tools, and assistant execution permissions are unchanged.

## Manual smoke test

1. Open a project with an active dataset and choose **Pipelines**.
2. Select **New Pipeline**, then **Save New Revision**.
3. Edit the JSON node list using the documented schema and one of the supported operation connections; use the active profile's dataset ID and actual sample bounds.
4. Select **Validate**. Resolve invalid schemas, metadata blockers and unavailable dataset bindings.
5. Save the edited definition as another revision, then select **Start Pipeline Run**.
6. Select **Prepare Next Node**, review the displayed request, and explicitly approve that node.
7. Repeat prepare/review/approve for each eligible downstream node. Confirm that failed dependencies block descendants while independent nodes remain available.
8. Close and reopen the project. Open the pipeline to inspect its saved revision and run summaries; no operation should execute until a new explicit run and approval.

