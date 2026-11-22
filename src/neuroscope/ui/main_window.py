from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..analysis.contracts import (
    DataReference,
    MetadataSnapshot,
    OperationContext,
    ReproducibilityInfo,
    SampleInterval,
)
from ..analysis.isi_operation import (
    ISIInput,
    ISIOutput,
    ISIParameters,
    isi_collection_from_record,
    register_isi_analysis,
)
from ..analysis.mua_average_operation import (
    AVERAGE_MUA_OPERATION,
    MUAAverageInput,
    MUAAverageOutput,
    MUAAverageParameters,
    mua_artifact_reference_from_record,
    mua_artifact_reference_from_snapshot,
    register_mua_average,
)
from ..analysis.mua_operation import OPERATION_ID as CALC_MUA_ID
from ..analysis.mua_operation import SOURCE_SHA256 as CALC_MUA_SOURCE_SHA256
from ..analysis.mua_operation import (
    CalcMUAInput,
    CalcMUAOutput,
    CalcMUAParameters,
    register_calc_mua,
)
from ..analysis.operations import (
    ApprovedExecution,
    ExecutionRecord,
    ExecutionStatus,
    OperationRunner,
    PreparedExecution,
)
from ..analysis.pipelines import (
    PipelineDefinition,
    PipelineExecutionService,
    PipelineRunState,
    new_pipeline,
    validate_pipeline,
)
from ..analysis.registry import OperationRegistry
from ..analysis.signed_peak_operation import OPERATION_ID as SIGNED_PEAK_ID
from ..analysis.signed_peak_operation import (
    SignedPeakInput,
    SignedPeakParameters,
    register_signed_peak_detector,
)
from ..analysis.spike_operation import OPERATION_ID as NEGATIVE_DETECTOR_ID
from ..analysis.spike_operation import (
    NegativeSpikeInput,
    NegativeSpikeParameters,
    register_negative_spike_detector,
)
from ..analysis.waveform_operation import (
    WaveformExtractionInput,
    WaveformExtractionOutput,
    WaveformExtractionParameters,
    detection_collection_from_record,
    register_waveform_extraction,
)
from ..artifacts import ISIArtifactStore, WaveformArtifactStore
from ..assistant.contracts import AssistantProposal, ChatMessage
from ..assistant.knowledge_packs import (
    KnowledgePackResource,
    KnowledgePackService,
)
from ..assistant.library import LibraryDocumentMetadata, LiteratureLibraryStore
from ..assistant.ollama_provider import (
    DEFAULT_OLLAMA_HOST,
    OllamaProvider,
    OllamaServiceStatus,
)
from ..assistant.orchestration import AssistantOrchestrator
from ..assistant.privacy import scrub_sensitive_text
from ..assistant.profiles import DEFAULT_PROFILE_ID, builtin_profiles
from ..assistant.providers import OpenAIProvider
from ..assistant.tools import AssistantApplicationState, AssistantToolService
from ..data_access import DatasetAccessError, DatasetRegistration, DatasetStore
from ..dataset_loading import (
    PreparedDataset,
    metadata_panel_text,
    plot_axis_labels,
    prepare_dataset,
)
from ..exploration import DatasetExploration, MetadataStatus
from ..memory.project_store import ProjectStore
from ..mua_artifacts import MUAArtifactStore
from ..navigation import NavigationState
from ..rendering import (
    RenderedSignal,
    read_original_sample,
    render_exploration_signal,
    sample_index_from_coordinate,
)


class _OperationWorker(QObject):
    """Run one already approved operation without touching any GUI object."""

    finished = Signal(int, object, object)

    def __init__(
        self,
        runner: OperationRunner,
        approved: ApprovedExecution,
        dataset_generation: int,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._approved = approved
        self._dataset_generation = dataset_generation

    @Slot()
    def run(self) -> None:
        try:
            record = self._runner.execute(self._approved)
            self.finished.emit(self._dataset_generation, record, None)
        except Exception as exc:  # noqa: BLE001 - return unexpected worker failures to the UI.
            self.finished.emit(self._dataset_generation, None, exc)


class _PipelineNodeWorker(QObject):
    """Execute one already approved pipeline node away from the GUI thread."""

    finished = Signal(object, object, object)

    def __init__(self, service, pipeline, state, node_id, approved, project_store):
        super().__init__()
        self._service = service
        self._pipeline = pipeline
        self._state = state
        self._node_id = node_id
        self._approved = approved
        self._project_store = project_store

    @Slot()
    def run(self) -> None:
        try:
            state, record = self._service.execute_node(
                self._pipeline, self._state, self._node_id, self._approved,
                project_store=self._project_store,
            )
            self.finished.emit(state, record, None)
        except Exception as exc:  # noqa: BLE001 - surface failures to GUI thread.
            self.finished.emit(None, None, exc)


class _OperationBatchWorker(QObject):
    """Execute individually approved operations in order, retaining partial results."""

    finished = Signal(int, object, object)

    def __init__(self, runner: OperationRunner, approvals, dataset_generation: int) -> None:
        super().__init__()
        self._runner = runner
        self._approvals = tuple(approvals)
        self._dataset_generation = dataset_generation

    @Slot()
    def run(self) -> None:
        records = []
        failure = None
        for approved in self._approvals:
            try:
                records.append(self._runner.execute(approved))
            except Exception as exc:  # noqa: BLE001 - partial batch is returned for review.
                failure = exc
                break
        self.finished.emit(self._dataset_generation, records, failure)


class _ViewWorker(QObject):
    """Perform bounded artifact reads away from the GUI thread."""

    finished = Signal(str, int, object, object)

    def __init__(self, view_name: str, generation: int, task) -> None:
        super().__init__()
        self._view_name = view_name
        self._generation = generation
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            result = self._task()
            self.finished.emit(self._view_name, self._generation, result, None)
        except Exception as exc:  # noqa: BLE001 - surface artifact failures in the UI.
            self.finished.emit(self._view_name, self._generation, None, exc)


class _AssistantWorker(QObject):
    """Run provider and validated local-tool orchestration outside Qt's UI thread."""

    finished = Signal(int, object, object)

    def __init__(self, orchestrator, history, generation: int, cancel_event: Event) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._history = tuple(history)
        self._generation = generation
        self._cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            turn = self._orchestrator.respond(
                self._history, cancel_event=self._cancel_event
            )
            self.finished.emit(self._generation, turn, None)
        except Exception as exc:  # noqa: BLE001 - surface provider failures without blocking UI.
            self.finished.emit(self._generation, None, exc)


class _OllamaStatusWorker(QObject):
    """Inspect a user-selected Ollama endpoint outside the GUI thread."""

    finished = Signal(int, object, object)

    def __init__(self, provider: OllamaProvider, generation: int) -> None:
        super().__init__()
        self._provider = provider
        self._generation = generation

    @Slot()
    def run(self) -> None:
        try:
            status = self._provider.inspect_service(timeout_s=min(self._provider.timeout_s, 8.0))
            self.finished.emit(self._generation, status, None)
        except Exception as exc:  # noqa: BLE001 - status errors are displayed safely.
            self.finished.emit(self._generation, None, exc)


class _LiteratureImportWorker(QObject):
    """Extract explicitly selected local sources away from the Qt GUI thread."""

    finished = Signal(object, object, int, object)

    def __init__(self, store, library_id, paths, retain_original, generation):
        super().__init__()
        self._store = store
        self._library_id = library_id
        self._paths = tuple(paths)
        self._retain_original = retain_original
        self._generation = generation

    @Slot()
    def run(self):
        imported = 0
        failures = []
        for path in self._paths:
            try:
                self._store.import_document(
                    self._library_id,
                    path,
                    metadata=LibraryDocumentMetadata(),
                    assistant_visible=False,
                    retain_original=self._retain_original,
                )
                imported += 1
            except Exception as exc:  # noqa: BLE001 - report per-file local ingestion failures.
                failures.append(f"{Path(path).name}: {scrub_sensitive_text(str(exc), limit=400)}")
        self.finished.emit(self._store, self._library_id, self._generation, (imported, tuple(failures)))


def _read_waveform_view(
    artifact_root: Path,
    artifact_reference: str,
    channel_index: int,
    maximum_traces: int,
    selected_event_id: str | None,
) -> dict[str, object]:
    """Read only a bounded deterministic waveform subset from its memmapped artifact."""
    with WaveformArtifactStore(artifact_root).open(artifact_reference) as artifact:
        manifest = artifact.manifest
        accepted = manifest.get("accepted_events", [])
        excluded = manifest.get("excluded_events", [])
        channel_events = [
            event for event in accepted if int(event["channel_index"]) == channel_index
        ]
        if selected_event_id:
            selected = next(
                (event for event in channel_events if event["event_id"] == selected_event_id),
                None,
            )
            if selected is None:
                raise ValueError(f"waveform event {selected_event_id!r} is unavailable on channel {channel_index}.")
        else:
            selected = channel_events[0] if channel_events else None
        if channel_events:
            local_positions = np.linspace(
                0,
                len(channel_events) - 1,
                num=min(maximum_traces, len(channel_events)),
                dtype=np.int64,
            )
            display_events = [channel_events[int(position)] for position in local_positions]
            display_ids = {event["event_id"] for event in display_events}
            if selected is not None and selected["event_id"] not in display_ids:
                display_events[-1] = selected
            rows = [int(event["waveform_row"]) for event in display_events]
            display_waveforms = np.asarray(artifact.waveforms[rows, :]).copy()
            selected_waveform = np.asarray(
                artifact.waveforms[int(selected["waveform_row"]), :]
            ).copy() if selected is not None else None
            summaries = {
                name: np.asarray(value).copy()
                for name, value in artifact.channel_summary(channel_index).items()
            }
        else:
            display_events = []
            display_waveforms = np.empty((0, int(manifest["waveform_shape"][1])))
            selected_waveform = None
            summaries = {}
        return {
            "manifest": manifest,
            "accepted_events": accepted,
            "excluded_events": excluded,
            "channel_events": channel_events,
            "display_events": display_events,
            "display_waveforms": display_waveforms,
            "selected_event": selected,
            "selected_waveform": selected_waveform,
            "summaries": summaries,
            "channel_index": channel_index,
        }


def _read_isi_view(
    artifact_root: Path,
    artifact_reference: str,
    channel_index: int,
    polarity_group: str,
    time_unit: str,
    bin_count: int,
    page_start: int,
    page_size: int = 100,
) -> dict[str, object]:
    """Stream an ISI JSONL artifact twice with bounded histogram/page memory."""
    with ISIArtifactStore(artifact_root).open(artifact_reference) as artifact:
        count = 0
        minimum = maximum = None
        for row in artifact.iter_rows():
            if row["channel_index"] != channel_index or row["polarity_group"] != polarity_group:
                continue
            value = row["delta_ms"] if time_unit == "milliseconds" else row["delta_samples"]
            if value is None:
                continue
            numeric = float(value)
            minimum = numeric if minimum is None else min(minimum, numeric)
            maximum = numeric if maximum is None else max(maximum, numeric)
            count += 1
        if count == 0:
            edges = np.asarray([], dtype=np.float64)
            counts = np.asarray([], dtype=np.int64)
        else:
            lower, upper = float(minimum), float(maximum)
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            edges = np.linspace(lower, upper, bin_count + 1)
            counts = np.zeros(bin_count, dtype=np.int64)
            for row in artifact.iter_rows():
                if row["channel_index"] != channel_index or row["polarity_group"] != polarity_group:
                    continue
                value = row["delta_ms"] if time_unit == "milliseconds" else row["delta_samples"]
                if value is None:
                    continue
                bucket = int(np.searchsorted(edges, float(value), side="right") - 1)
                bucket = min(max(bucket, 0), bin_count - 1)
                counts[bucket] += 1
        page: list[dict[str, object]] = []
        seen = 0
        for row in artifact.iter_rows():
            if row["channel_index"] != channel_index or row["polarity_group"] != polarity_group:
                continue
            if page_start <= seen < page_start + page_size:
                page.append(row)
            seen += 1
            if seen >= page_start + page_size:
                break
        return {
            "histogram_edges": edges,
            "histogram_counts": counts,
            "interval_count": count,
            "page_rows": page,
            "page_start": page_start,
        }


def _read_mua_view(
    artifact_root: Path,
    artifact_reference: str,
    exploration: DatasetExploration | None,
    channel_index: int,
    maximum_points: int,
) -> dict[str, object]:
    """Read a bounded MUA trace view and matching source-signal envelope."""
    with MUAArtifactStore(artifact_root).open(artifact_reference) as artifact:
        manifest = artifact.manifest
        interval = manifest["sample_interval"]
        start = int(interval["start"])
        stop = int(interval["stop"])
        trace = artifact.mua
        bucket_width = max(1, (trace.size + maximum_points - 1) // maximum_points)
        trace_indices: list[int] = []
        trace_values: list[float] = []
        for bucket_start in range(0, trace.size, bucket_width):
            bucket_stop = min(bucket_start + bucket_width, trace.size)
            block = trace[bucket_start:bucket_stop]
            finite_offsets = np.flatnonzero(np.isfinite(block))
            if not finite_offsets.size:
                continue
            finite_values = block[finite_offsets]
            extrema = sorted(
                {
                    int(finite_offsets[int(np.argmin(finite_values))]),
                    int(finite_offsets[int(np.argmax(finite_values))]),
                }
            )
            for offset in extrema:
                trace_indices.append(start + bucket_start + offset)
                trace_values.append(float(block[offset]))

        if exploration is None:
            source_indices = np.empty(0, dtype=np.int64)
            source_values = np.empty(0, dtype=np.float64)
        else:
            rendered = render_exploration_signal(
                exploration,
                channel_index=channel_index,
                start_sample=start,
                stop_sample=stop,
                pixel_width=maximum_points,
            )
            source_indices = rendered.sample_indices
            source_values = rendered.values
        return {
            "manifest": manifest,
            "trace_sample_indices": np.asarray(trace_indices, dtype=np.int64),
            "trace_values": np.asarray(trace_values, dtype=np.float64),
            "source_sample_indices": source_indices,
            "source_values": source_values,
        }


def _read_mua_selection(
    artifact_root: Path,
    entries: tuple[dict[str, object], ...],
    maximum_points: int,
) -> list[dict[str, object]]:
    """Read selected traces with bounded display memory, off the GUI thread."""
    results = []
    for entry in entries:
        with MUAArtifactStore(artifact_root).open(str(entry["artifact_reference"])) as artifact:
            trace = artifact.mua
            start = int(entry["sample_start"])
            maximum = max(1, maximum_points // max(1, len(entries)))
            bucket_width = max(1, (trace.size + maximum - 1) // maximum)
            indices: list[int] = []
            values: list[float] = []
            for bucket_start in range(0, trace.size, bucket_width):
                block = trace[bucket_start : bucket_start + bucket_width]
                finite = np.flatnonzero(np.isfinite(block))
                if not finite.size:
                    continue
                local = block[finite]
                for offset in sorted(
                    {int(finite[np.argmin(local)]), int(finite[np.argmax(local)])}
                ):
                    indices.append(start + bucket_start + offset)
                    values.append(float(block[offset]))
            results.append(
                {
                    "execution_id": entry["execution_id"],
                    "label": entry["label"],
                    "indices": np.asarray(indices, dtype=np.int64),
                    "values": np.asarray(values, dtype=np.float64),
                    "sampling_rate_hz": entry["sampling_rate_hz"],
                    "sampling_rate_status": entry["sampling_rate_status"],
                    "units": entry["units"],
                    "mode": entry["mode"],
                    "interval": (start, start + trace.size),
                    "artifact_type": artifact.manifest.get("artifact_type", "individual_calc_mua"),
                }
            )
    return results


class _FullRecordingToolbar(NavigationToolbar2QT):
    """Matplotlib toolbar whose Home action always restores the full recording."""

    def __init__(self, canvas, parent, reset_callback):
        self._reset_callback = reset_callback
        super().__init__(canvas, parent)

    def home(self, *args):
        self._reset_callback()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScope AI — Neural Data Analysis")
        self.resize(1280, 800)
        self.active_dataset: PreparedDataset | None = None
        self.active_registration: DatasetRegistration | None = None
        self.navigation_state: NavigationState | None = None
        self.dataset_store = DatasetStore()
        self.operation_registry = OperationRegistry()
        register_negative_spike_detector(self.operation_registry)
        register_signed_peak_detector(self.operation_registry)
        register_waveform_extraction(self.operation_registry)
        register_isi_analysis(self.operation_registry)
        register_calc_mua(self.operation_registry)
        register_mua_average(self.operation_registry)
        self.operation_runner = OperationRunner(
            self.operation_registry,
            data_store=self.dataset_store,
        )
        self.pipeline_service = PipelineExecutionService(
            self.operation_registry, self.operation_runner, self.dataset_store
        )
        self._pipeline_run_state: PipelineRunState | None = None
        self._pipeline_prepared = None
        self._pipeline_worker_thread: QThread | None = None
        self._pipeline_worker: _PipelineNodeWorker | None = None
        self._prepared_execution: PreparedExecution | None = None
        self._prepared_mua_batch: tuple[PreparedExecution, ...] | None = None
        self._current_record: ExecutionRecord | None = None
        self._current_events: list[dict[str, object]] = []
        self._execution_records: list[ExecutionRecord] = []
        self._detector_records: dict[str, ExecutionRecord] = {}
        self._current_waveform_record: ExecutionRecord | None = None
        self._current_isi_record: ExecutionRecord | None = None
        self._current_mua_record: ExecutionRecord | None = None
        self._mua_records: dict[str, ExecutionRecord] = {}
        self._mua_outputs: dict[str, CalcMUAOutput | MUAAverageOutput] = {}
        self._mua_input_references = {}
        self._mua_provenance_snapshots: dict[str, dict[str, object]] = {}
        self._waveform_artifact_view: dict[str, object] | None = None
        self._isi_artifact_view: dict[str, object] | None = None
        self._mua_artifact_view: dict[str, object] | None = None
        self._mua_selection_views: list[dict[str, object]] = []
        self._view_generations = {"waveform": 0, "isi": 0, "mua": 0}
        self._view_workers: list[tuple[QThread, _ViewWorker]] = []
        self._result_stale = False
        self._dataset_generation = 0
        self._worker_thread: QThread | None = None
        self._worker: _OperationWorker | None = None
        self._execution_in_flight = False
        self._pending_save_record: ExecutionRecord | None = None
        self._assistant_settings = QSettings("NeuroScope AI", "NeuroScope AI")
        self._scientific_profiles = builtin_profiles()
        configured_profile_id = str(
            self._assistant_settings.value("assistant/scientific_profile", DEFAULT_PROFILE_ID)
        )
        try:
            self._assistant_scientific_profile = self._scientific_profiles.get(configured_profile_id)
        except KeyError:
            self._assistant_scientific_profile = self._scientific_profiles.get(DEFAULT_PROFILE_ID)
        self._assistant_provider_kind = str(
            self._assistant_settings.value("assistant/provider", "openai")
        )
        if self._assistant_provider_kind not in {"openai", "ollama"}:
            self._assistant_provider_kind = "openai"
        ollama_defaults = OllamaProvider()
        self._ollama_host = str(
            self._assistant_settings.value("assistant/ollama_host", ollama_defaults.host)
        )
        self._ollama_model = str(
            self._assistant_settings.value("assistant/ollama_model", ollama_defaults.model)
        )
        self._ollama_status: OllamaServiceStatus | None = None
        self._ollama_status_generation = 0
        self._ollama_status_thread: QThread | None = None
        self._ollama_status_worker: _OllamaStatusWorker | None = None
        self._assistant_provider = self._make_assistant_provider()
        self._assistant_tools = AssistantToolService(self._assistant_state)
        self._assistant_conversation: list[ChatMessage] = []
        self._assistant_generation = 0
        self._assistant_cancelled_generations: set[int] = set()
        self._assistant_thread: QThread | None = None
        self._assistant_worker: _AssistantWorker | None = None
        self._assistant_cancel_event: Event | None = None
        self._assistant_last_question: str | None = None
        self._assistant_approval_proposal_id: str | None = None
        self._assistant_approval_ids: set[str] = set()
        self._assistant_external_document_allowlist: tuple[str, ...] = ()
        self._literature_import_thread: QThread | None = None
        self._literature_import_worker: _LiteratureImportWorker | None = None
        self._updating_plot = False
        self._inspecting_event = False
        self._linking_event = False
        self._render_cache_key: tuple[object, ...] | None = None
        self._rendered_visible_window = None
        self._render_cache: dict[int, RenderedSignal] = {}
        self._plot_channel_by_axis = {}
        self._cursor_lines = {}
        self._event_scatters = {}
        # A store is assigned only by an explicit project action, never on load.
        self.store: ProjectStore | None = None
        self._literature_store: LiteratureLibraryStore | None = None
        self._build_ui()
        self._refresh_literature_libraries()
        self._update_assistant_provider_status()
        self._update_detector_controls()
        self._update_action_state()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        self.load_button = QPushButton("Load dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.project_button = QPushButton("Select / create project")
        self.project_button.clicked.connect(self.select_project)
        self.open_project_button = QPushButton("Open project")
        self.open_project_button.clicked.connect(self.open_project)
        self.project_label = QLabel("No dataset loaded")
        self.project_path_label = QLabel("No project selected")
        top.addWidget(self.load_button)
        top.addWidget(self.project_label, 1)
        top.addWidget(self.open_project_button)
        top.addWidget(self.project_button)
        layout.addLayout(top)
        project_row = QHBoxLayout()
        project_row.addWidget(QLabel("Project storage:"))
        project_row.addWidget(self.project_path_label, 1)
        layout.addLayout(project_row)

        splitter = QSplitter(Qt.Horizontal)
        left_content = QWidget(); left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(QLabel("Dataset information"))
        self.profile_view = QTextEdit(); self.profile_view.setReadOnly(True)
        left_layout.addWidget(self.profile_view)
        left_layout.addWidget(QLabel("Displayed channels (0-based original indices)"))
        self.display_channel_list = QListWidget()
        self.display_channel_list.setVisible(False)
        self.display_channel_list.itemChanged.connect(self._on_display_channels_changed)
        left_layout.addWidget(self.display_channel_list)
        display_actions = QHBoxLayout()
        self.select_all_channels_button = QPushButton("Select all")
        self.select_all_channels_button.setVisible(False)
        self.select_all_channels_button.clicked.connect(
            lambda: self._set_all_display_channels(True)
        )
        self.clear_channels_button = QPushButton("Clear")
        self.clear_channels_button.setVisible(False)
        self.clear_channels_button.clicked.connect(
            lambda: self._set_all_display_channels(False)
        )
        display_actions.addWidget(self.select_all_channels_button)
        display_actions.addWidget(self.clear_channels_button)
        left_layout.addLayout(display_actions)
        left_layout.addWidget(QLabel("Interactive spike lab"))
        self.detector_box = QComboBox()
        self.detector_box.addItem("Negative MAD detector", NEGATIVE_DETECTOR_ID)
        self.detector_box.addItem("Signed-peak detector", SIGNED_PEAK_ID)
        self.channel_box = QComboBox(); self.channel_box.addItem("Select channel")
        self.threshold = QLineEdit("-4")
        self.threshold.setPlaceholderText("Negative threshold_sd")
        self.signed_mode_box = QComboBox()
        self.signed_mode_box.addItem("Absolute signed threshold", "absolute")
        self.signed_mode_box.addItem("Signed noise multiplier", "noise_multiplier")
        self.signed_threshold = QLineEdit()
        self.signed_threshold.setPlaceholderText("Signed threshold / multiplier")
        self.noise_override = QLineEdit()
        self.noise_override.setPlaceholderText("Optional noise override")
        self.polarity_label = QLabel("Polarity: negative")
        self.detector_explanation = QLabel()
        self.detector_explanation.setWordWrap(True)
        self.prepare_button = QPushButton("Prepare Analysis")
        self.prepare_button.clicked.connect(self.prepare_spike_detection)
        self.approve_button = QPushButton("Approve & Run")
        self.approve_button.clicked.connect(self.approve_and_run)
        self.save_record_button = QPushButton("Retry saving execution record")
        self.save_record_button.clicked.connect(self.retry_save_execution_record)
        self.save_record_button.setVisible(False)
        self.analysis_state_label = QLabel("No dataset")
        self.operation_review = QTextEdit()
        self.operation_review.setReadOnly(True)
        self.operation_review.setPlaceholderText("Prepare an operation to review its inputs and warnings.")
        left_layout.addWidget(QLabel("Channel")); left_layout.addWidget(self.channel_box)
        left_layout.addWidget(QLabel("Detector")); left_layout.addWidget(self.detector_box)
        left_layout.addWidget(self.detector_explanation)
        self.negative_controls = QWidget()
        negative_layout = QVBoxLayout(self.negative_controls)
        negative_layout.setContentsMargins(0, 0, 0, 0)
        negative_layout.addWidget(QLabel("threshold_sd (negative MAD multiplier; dimensionless)"))
        negative_layout.addWidget(self.threshold)
        left_layout.addWidget(self.negative_controls)
        self.signed_controls = QWidget()
        signed_layout = QVBoxLayout(self.signed_controls)
        signed_layout.setContentsMargins(0, 0, 0, 0)
        signed_layout.addWidget(QLabel("Signed-peak threshold mode"))
        signed_layout.addWidget(self.signed_mode_box)
        signed_layout.addWidget(QLabel("Signed parameter (signal units for absolute mode; dimensionless for multiplier)"))
        signed_layout.addWidget(self.signed_threshold)
        signed_layout.addWidget(QLabel("Noise override (amplitude units; absolute mode only)"))
        signed_layout.addWidget(self.noise_override)
        signed_layout.addWidget(self.polarity_label)
        left_layout.addWidget(self.signed_controls)
        actions = QHBoxLayout()
        actions.addWidget(self.prepare_button)
        actions.addWidget(self.approve_button)
        left_layout.addLayout(actions)
        left_layout.addWidget(self.analysis_state_label)
        left_layout.addWidget(self.operation_review, 1)
        left_layout.addWidget(QLabel("Detected events (select a row for details)"))
        self.event_table = QTableWidget(0, 5)
        self.event_table.setHorizontalHeaderLabels(
            ["Event", "Sample", "Channel", "Polarity", "Amplitude"]
        )
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.event_table.cellClicked.connect(self._select_detected_event)
        left_layout.addWidget(self.event_table)
        self.event_details = QTextEdit()
        self.event_details.setReadOnly(True)
        self.event_details.setMaximumHeight(110)
        left_layout.addWidget(self.event_details)

        left_layout.addWidget(QLabel("Waveform Explorer"))
        self.waveform_saved_box = QComboBox()
        self.waveform_saved_box.addItem("Select saved waveform artifact", None)
        self.open_saved_waveform_button = QPushButton("Open saved waveform result")
        self.open_saved_waveform_button.clicked.connect(self.open_saved_waveform_result)
        left_layout.addWidget(self.waveform_saved_box)
        left_layout.addWidget(self.open_saved_waveform_button)
        self.waveform_source_box = QComboBox()
        self.waveform_source_box.addItem("Select detector execution", None)
        self.waveform_channel_filter = QComboBox()
        self.waveform_channel_filter.addItem("All detected channels", None)
        self.waveform_only_selected_event = QCheckBox("Extract only the selected event")
        self.waveform_pre_samples = QLineEdit()
        self.waveform_pre_samples.setPlaceholderText("Pre-event samples (required)")
        self.waveform_post_samples = QLineEdit()
        self.waveform_post_samples.setPlaceholderText("Post-event samples (required)")
        self.waveform_length_label = QLabel("Waveform length: — samples")
        self.prepare_waveforms_button = QPushButton("Prepare Waveform Extraction")
        self.prepare_waveforms_button.clicked.connect(self.prepare_waveform_extraction)
        for widget in (
            self.waveform_source_box,
            self.waveform_channel_filter,
            self.waveform_only_selected_event,
            self.waveform_pre_samples,
            self.waveform_post_samples,
            self.waveform_length_label,
            self.prepare_waveforms_button,
        ):
            left_layout.addWidget(widget)

        left_layout.addWidget(QLabel("ISI Explorer"))
        self.isi_saved_box = QComboBox()
        self.isi_saved_box.addItem("Select saved ISI artifact", None)
        self.open_saved_isi_button = QPushButton("Open saved ISI result")
        self.open_saved_isi_button.clicked.connect(self.open_saved_isi_result)
        left_layout.addWidget(self.isi_saved_box)
        left_layout.addWidget(self.open_saved_isi_button)
        self.isi_source_box = QComboBox()
        self.isi_source_box.addItem("Select detector execution", None)
        self.isi_polarity_box = QComboBox()
        self.isi_polarity_box.addItem("Select polarity grouping", None)
        self.isi_polarity_box.addItem("Separate polarities", "separate")
        self.isi_polarity_box.addItem("Combine within channel", "combine")
        self.isi_threshold_enabled = QCheckBox("Calculate short-interval metrics")
        self.isi_threshold = QLineEdit()
        self.isi_threshold.setPlaceholderText("Optional explicit threshold")
        self.isi_threshold.setEnabled(False)
        self.isi_threshold_units = QComboBox()
        self.isi_threshold_units.addItem("Samples", "samples")
        self.isi_threshold_units.addItem("Milliseconds", "milliseconds")
        self.isi_threshold_units.setEnabled(False)
        self.prepare_isi_button = QPushButton("Prepare ISI Analysis")
        self.prepare_isi_button.clicked.connect(self.prepare_isi_analysis)
        for widget in (
            self.isi_source_box,
            self.isi_polarity_box,
            self.isi_threshold_enabled,
            self.isi_threshold,
            self.isi_threshold_units,
            self.prepare_isi_button,
        ):
            left_layout.addWidget(widget)
        left_layout.addWidget(QLabel("MUAtrace — original calcMUA"))
        left_layout.addWidget(QLabel("Channels to calculate (independent per-channel operations)"))
        self.mua_calculation_channels = QListWidget()
        self.mua_calculation_channels.setMaximumHeight(95)
        left_layout.addWidget(self.mua_calculation_channels)
        # Kept as an invisible compatibility selector for existing single-channel UI callers.
        self.mua_channel_box = QComboBox()
        self.mua_channel_box.addItem("Select one source channel", None)
        self.mua_stdmin = QLineEdit()
        self.mua_stdmin.setPlaceholderText("Signed stdmin (required)")
        self.mua_bandwidth = QLineEdit()
        self.mua_bandwidth.setPlaceholderText("muawin bandwidth in seconds (required)")
        self.mua_mode_box = QComboBox()
        self.mua_mode_box.addItem("Select output mode", None)
        self.mua_mode_box.addItem("Relative density (rel)", "rel")
        self.mua_mode_box.addItem("Count-scaled density (abs)", "abs")
        self.mua_num_points = QLineEdit()
        self.mua_num_points.setPlaceholderText("num_points (required; source grid is explicit)")
        self.prepare_mua_button = QPushButton("Prepare calcMUA")
        self.prepare_mua_button.clicked.connect(self.prepare_calc_mua)
        for widget in (
            self.mua_stdmin,
            self.mua_bandwidth,
            self.mua_mode_box,
            self.mua_num_points,
            self.prepare_mua_button,
        ):
            left_layout.addWidget(widget)
        left_layout.addWidget(QLabel("Existing MUAtraces to display (visibility only)"))
        self.mua_display_results = QListWidget()
        self.mua_display_results.setMaximumHeight(105)
        left_layout.addWidget(self.mua_display_results)
        left_layout.addWidget(QLabel("MUAtraces to average (explicit scientific inputs)"))
        self.mua_average_inputs = QListWidget()
        self.mua_average_inputs.setMaximumHeight(105)
        left_layout.addWidget(self.mua_average_inputs)
        self.prepare_mua_average_button = QPushButton("Prepare Average")
        self.prepare_mua_average_button.clicked.connect(self.prepare_mua_average)
        left_layout.addWidget(self.prepare_mua_average_button)
        self.waveform_source_box.currentIndexChanged.connect(self._on_waveform_source_changed)
        self.waveform_channel_filter.currentIndexChanged.connect(self._on_waveform_selection_changed)
        self.waveform_only_selected_event.stateChanged.connect(self._on_waveform_selection_changed)
        self.waveform_pre_samples.textChanged.connect(self._on_waveform_parameter_changed)
        self.waveform_post_samples.textChanged.connect(self._on_waveform_parameter_changed)
        self.isi_source_box.currentIndexChanged.connect(self._on_isi_source_changed)
        self.isi_polarity_box.currentIndexChanged.connect(self._on_isi_parameters_changed)
        self.isi_threshold_enabled.stateChanged.connect(self._on_isi_threshold_toggled)
        self.isi_threshold.textChanged.connect(self._on_isi_parameters_changed)
        self.isi_threshold_units.currentIndexChanged.connect(self._on_isi_parameters_changed)
        self.mua_channel_box.currentIndexChanged.connect(self._on_mua_parameters_changed)
        self.mua_stdmin.textChanged.connect(self._on_mua_parameters_changed)
        self.mua_bandwidth.textChanged.connect(self._on_mua_parameters_changed)
        self.mua_mode_box.currentIndexChanged.connect(self._on_mua_parameters_changed)
        self.mua_num_points.textChanged.connect(self._on_mua_parameters_changed)
        self.mua_calculation_channels.itemChanged.connect(self._on_mua_parameters_changed)
        self.mua_display_results.itemChanged.connect(self._on_mua_display_changed)
        self.mua_average_inputs.itemChanged.connect(self._on_mua_average_selection_changed)

        self.detector_box.currentIndexChanged.connect(self._on_detector_changed)
        self.channel_box.currentIndexChanged.connect(self._on_analysis_channel_changed)
        self.threshold.textChanged.connect(self._on_parameters_changed)
        self.signed_threshold.textChanged.connect(self._on_parameters_changed)
        self.noise_override.textChanged.connect(self._on_parameters_changed)
        self.signed_mode_box.currentIndexChanged.connect(self._on_signed_mode_changed)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.workspace_tabs = QTabWidget()
        signal_page = QWidget(); signal_layout = QVBoxLayout(signal_page)
        self.figure = Figure(figsize=(7, 4)); self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_cursor_motion)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self.navigation_toolbar = _FullRecordingToolbar(
            self.canvas, self, self.reset_navigation_view
        )
        signal_layout.addWidget(self.navigation_toolbar)
        signal_layout.addWidget(self.canvas, 3)
        self.workspace_tabs.addTab(signal_page, "Signal & detections")

        waveform_page = QWidget(); waveform_layout = QVBoxLayout(waveform_page)
        waveform_controls = QHBoxLayout()
        self.waveform_event_box = QComboBox()
        self.waveform_event_box.setEditable(True)
        self.waveform_event_box.setMinimumWidth(220)
        self.waveform_channel_box = QComboBox()
        self.waveform_max_display = QSpinBox()
        self.waveform_max_display.setRange(1, 500)
        self.waveform_max_display.setValue(25)
        self.waveform_show_individual = QCheckBox("Individual")
        self.waveform_show_individual.setChecked(True)
        self.waveform_show_mean = QCheckBox("Mean")
        self.waveform_show_mean.setChecked(True)
        self.waveform_show_median = QCheckBox("Median")
        self.waveform_show_std = QCheckBox("Mean ± SD")
        for label, widget in (
            ("Event", self.waveform_event_box),
            ("Channel", self.waveform_channel_box),
            ("Max traces", self.waveform_max_display),
        ):
            waveform_controls.addWidget(QLabel(label)); waveform_controls.addWidget(widget)
        for widget in (
            self.waveform_show_individual,
            self.waveform_show_mean,
            self.waveform_show_median,
            self.waveform_show_std,
        ):
            waveform_controls.addWidget(widget)
        waveform_layout.addLayout(waveform_controls)
        self.waveform_display_label = QLabel("No waveform extraction selected.")
        waveform_layout.addWidget(self.waveform_display_label)
        self.waveform_figure = Figure(figsize=(7, 4))
        self.waveform_canvas = FigureCanvasQTAgg(self.waveform_figure)
        waveform_layout.addWidget(self.waveform_canvas, 3)
        waveform_layout.addWidget(QLabel("Boundary-excluded events"))
        self.waveform_excluded_table = QTableWidget(0, 3)
        self.waveform_excluded_table.setHorizontalHeaderLabels(["Event", "Sample", "Reason"])
        self.waveform_excluded_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        waveform_layout.addWidget(self.waveform_excluded_table, 1)
        self.workspace_tabs.addTab(waveform_page, "Waveforms")

        isi_page = QWidget(); isi_layout = QVBoxLayout(isi_page)
        isi_controls = QHBoxLayout()
        self.isi_group_box = QComboBox()
        self.isi_histogram_bins = QSpinBox(); self.isi_histogram_bins.setRange(1, 500); self.isi_histogram_bins.setValue(50)
        self.isi_page_start = QSpinBox(); self.isi_page_start.setRange(0, 2_147_483_647)
        isi_controls.addWidget(QLabel("Channel / polarity group")); isi_controls.addWidget(self.isi_group_box)
        isi_controls.addWidget(QLabel("Histogram bins")); isi_controls.addWidget(self.isi_histogram_bins)
        isi_controls.addWidget(QLabel("Interval row offset")); isi_controls.addWidget(self.isi_page_start)
        isi_layout.addLayout(isi_controls)
        self.isi_display_label = QLabel("No ISI analysis selected.")
        isi_layout.addWidget(self.isi_display_label)
        self.isi_figure = Figure(figsize=(7, 3))
        self.isi_canvas = FigureCanvasQTAgg(self.isi_figure)
        isi_layout.addWidget(self.isi_canvas, 2)
        self.isi_interval_table = QTableWidget(0, 7)
        self.isi_interval_table.setHorizontalHeaderLabels(
            ["Start event", "End event", "Channel", "Group", "Start sample", "End sample", "ISI"]
        )
        self.isi_interval_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.isi_interval_table.cellClicked.connect(self._select_isi_interval)
        isi_layout.addWidget(self.isi_interval_table, 2)
        self.workspace_tabs.addTab(isi_page, "ISIs")
        mua_page = QWidget(); mua_layout = QVBoxLayout(mua_page)
        self.mua_display_label = QLabel("No calcMUA result selected.")
        mua_layout.addWidget(self.mua_display_label)
        mua_toolbar = QHBoxLayout()
        mua_toolbar.addWidget(QLabel("Trace layout"))
        self.mua_layout_box = QComboBox()
        self.mua_layout_box.addItem("Overlay", "overlay")
        self.mua_layout_box.addItem("Separate panels", "separate")
        self.mua_layout_box.currentIndexChanged.connect(self._on_mua_layout_changed)
        mua_toolbar.addWidget(self.mua_layout_box)
        self.mua_provenance_button = QPushButton("Inspect provenance")
        self.mua_provenance_button.clicked.connect(self._inspect_mua_provenance)
        mua_toolbar.addWidget(self.mua_provenance_button)
        mua_layout.addLayout(mua_toolbar)
        self.mua_figure = Figure(figsize=(7, 4))
        self.mua_canvas = FigureCanvasQTAgg(self.mua_figure)
        self.mua_canvas.mpl_connect("button_press_event", self._on_mua_plot_click)
        self.mua_canvas.mpl_connect("draw_event", self._sync_mua_navigation_limits)
        mua_layout.addWidget(self.mua_canvas, 1)
        self.workspace_tabs.addTab(mua_page, "MUAtrace")

        pipeline_page = QWidget(); pipeline_layout = QVBoxLayout(pipeline_page)
        pipeline_layout.addWidget(QLabel(
            "Pipeline Explorer — edit versioned JSON definitions. Every new scientific node "
            "is reviewed and approved separately. Opening a pipeline never runs it."
        ))
        self.pipeline_json_editor = QTextEdit()
        self.pipeline_json_editor.setPlaceholderText(
            "Create a pipeline, then add nodes using registered operation IDs and explicit input bindings."
        )
        pipeline_layout.addWidget(self.pipeline_json_editor, 3)
        pipeline_buttons = QVBoxLayout()
        pipeline_row_one = QHBoxLayout()
        pipeline_row_two = QHBoxLayout()
        pipeline_row_three = QHBoxLayout()
        self.pipeline_new_button = QPushButton("New Pipeline")
        self.pipeline_new_button.clicked.connect(self._new_pipeline_draft)
        self.pipeline_open_button = QPushButton("Open Saved Pipeline")
        self.pipeline_open_button.clicked.connect(self._open_saved_pipeline)
        self.pipeline_validate_button = QPushButton("Validate")
        self.pipeline_validate_button.clicked.connect(self._validate_pipeline_ui)
        self.pipeline_save_button = QPushButton("Save New Revision")
        self.pipeline_save_button.clicked.connect(self._save_pipeline_ui)
        self.pipeline_add_node_button = QPushButton("Add Registered Operation")
        self.pipeline_add_node_button.clicked.connect(self._pipeline_add_operation_node)
        self.pipeline_dataset_binding_button = QPushButton("Bind Active Dataset")
        self.pipeline_dataset_binding_button.clicked.connect(self._pipeline_add_dataset_binding)
        self.pipeline_connect_button = QPushButton("Connect Compatible Nodes")
        self.pipeline_connect_button.clicked.connect(self._pipeline_connect_nodes)
        self.pipeline_artifact_button = QPushButton("Select Saved MUA Artifact")
        self.pipeline_artifact_button.clicked.connect(self._pipeline_select_mua_artifact)
        self.pipeline_reuse_button = QPushButton("Reuse Saved Execution")
        self.pipeline_reuse_button.clicked.connect(self._pipeline_reuse_execution)
        self.pipeline_start_button = QPushButton("Start Pipeline Run")
        self.pipeline_start_button.clicked.connect(self._start_pipeline_ui)
        self.pipeline_prepare_button = QPushButton("Prepare Next Node")
        self.pipeline_prepare_button.clicked.connect(self._prepare_pipeline_node_ui)
        self.pipeline_approve_button = QPushButton("Approve & Execute Prepared Node")
        self.pipeline_approve_button.clicked.connect(self._approve_pipeline_node_ui)
        self.pipeline_dismiss_button = QPushButton("Dismiss Prepared Node")
        self.pipeline_dismiss_button.clicked.connect(self._dismiss_pipeline_node_ui)
        for button in (self.pipeline_new_button, self.pipeline_open_button, self.pipeline_validate_button, self.pipeline_save_button):
            pipeline_row_one.addWidget(button)
        for button in (self.pipeline_add_node_button, self.pipeline_dataset_binding_button, self.pipeline_connect_button, self.pipeline_artifact_button, self.pipeline_reuse_button):
            pipeline_row_two.addWidget(button)
        for button in (self.pipeline_start_button, self.pipeline_prepare_button, self.pipeline_approve_button, self.pipeline_dismiss_button):
            pipeline_row_three.addWidget(button)
        pipeline_buttons.addLayout(pipeline_row_one)
        pipeline_buttons.addLayout(pipeline_row_two)
        pipeline_buttons.addLayout(pipeline_row_three)
        pipeline_layout.addLayout(pipeline_buttons)
        self.pipeline_status = QTextEdit()
        self.pipeline_status.setReadOnly(True)
        self.pipeline_status.setMaximumHeight(180)
        self.pipeline_status.setPlaceholderText("Validation, node readiness, provenance, and execution status appear here.")
        pipeline_layout.addWidget(self.pipeline_status, 1)
        self.workspace_tabs.addTab(pipeline_page, "Pipelines")

        literature_page = QWidget(); literature_layout = QVBoxLayout(literature_page)
        literature_layout.addWidget(QLabel(
            "Local Literature Library — imported content stays in this project by default. "
            "Library scopes are organizational labels, not access control or encryption."
        ))
        library_controls = QHBoxLayout()
        self.literature_library_box = QListWidget()
        self.literature_library_box.setMaximumHeight(110)
        self.literature_name_input = QLineEdit()
        self.literature_name_input.setPlaceholderText("Library name")
        self.literature_scope_box = QComboBox()
        for label, value in (("Core", "core"), ("Personal", "personal"), ("Laboratory", "laboratory"), ("Project-specific", "project")):
            self.literature_scope_box.addItem(label, value)
        self.literature_create_button = QPushButton("Create library")
        self.literature_create_button.clicked.connect(self._create_literature_library)
        library_controls.addWidget(self.literature_library_box, 2)
        library_controls.addWidget(self.literature_name_input, 1)
        library_controls.addWidget(self.literature_scope_box)
        library_controls.addWidget(self.literature_create_button)
        literature_layout.addLayout(library_controls)
        library_actions = QHBoxLayout()
        self.literature_open_catalog_button = QPushButton("Open separate local catalog")
        self.literature_open_catalog_button.clicked.connect(self._open_literature_catalog)
        self.literature_import_button = QPushButton("Import PDF / TXT / Markdown")
        self.literature_import_button.clicked.connect(self._import_literature_documents)
        self.literature_remove_button = QPushButton("Remove selected document")
        self.literature_remove_button.clicked.connect(self._remove_literature_document)
        self.literature_inspect_button = QPushButton("Inspect metadata")
        self.literature_inspect_button.clicked.connect(self._inspect_literature_document)
        self.literature_edit_status_button = QPushButton("Set explicit status")
        self.literature_edit_status_button.clicked.connect(self._edit_literature_status)
        self.literature_replace_button = QPushButton("Replace selected document")
        self.literature_replace_button.clicked.connect(self._replace_literature_document)
        self.literature_manifest_export_button = QPushButton("Export metadata manifest")
        self.literature_manifest_export_button.clicked.connect(self._export_literature_manifest)
        self.literature_payload_export_button = QPushButton("Export authorized payloads")
        self.literature_payload_export_button.clicked.connect(self._export_literature_payloads)
        self.literature_manifest_import_button = QPushButton("Import metadata manifest")
        self.literature_manifest_import_button.clicked.connect(self._import_literature_manifest)
        for button in (self.literature_open_catalog_button, self.literature_import_button, self.literature_replace_button, self.literature_inspect_button, self.literature_edit_status_button, self.literature_remove_button, self.literature_manifest_export_button, self.literature_payload_export_button, self.literature_manifest_import_button):
            library_actions.addWidget(button)
        literature_layout.addLayout(library_actions)
        pack_actions = QHBoxLayout()
        self.knowledge_pack_export_button = QPushButton("Create / export Knowledge Pack from selected documents")
        self.knowledge_pack_export_button.clicked.connect(self._create_knowledge_pack)
        self.knowledge_pack_import_button = QPushButton("Inspect / import Knowledge Pack")
        self.knowledge_pack_import_button.clicked.connect(self._import_knowledge_pack)
        self.knowledge_pack_inspect_button = QPushButton("Inspect selected Knowledge Pack")
        self.knowledge_pack_inspect_button.clicked.connect(self._inspect_knowledge_pack)
        self.knowledge_pack_remove_button = QPushButton("Remove selected Knowledge Pack")
        self.knowledge_pack_remove_button.clicked.connect(self._remove_knowledge_pack)
        for button in (self.knowledge_pack_export_button, self.knowledge_pack_import_button, self.knowledge_pack_inspect_button, self.knowledge_pack_remove_button):
            pack_actions.addWidget(button)
        literature_layout.addLayout(pack_actions)
        self.literature_documents = QTableWidget(0, 6)
        self.literature_documents.setHorizontalHeaderLabels(["Assistant-visible", "Title", "DOI", "Publication status", "Access", "Extraction"])
        self.literature_documents.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.literature_documents.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.literature_documents.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.literature_documents.itemChanged.connect(self._on_literature_document_visibility_changed)
        literature_layout.addWidget(self.literature_documents, 2)
        search_row = QHBoxLayout()
        self.literature_query_input = QLineEdit()
        self.literature_query_input.setPlaceholderText("Search selected libraries")
        self.literature_search_button = QPushButton("Search local library")
        self.literature_search_button.clicked.connect(self._search_literature_library)
        self.literature_open_button = QPushButton("Open retained source")
        self.literature_open_button.clicked.connect(self._open_literature_source)
        search_row.addWidget(self.literature_query_input, 1)
        search_row.addWidget(self.literature_search_button)
        search_row.addWidget(self.literature_open_button)
        literature_layout.addLayout(search_row)
        self.literature_results = QTextEdit(); self.literature_results.setReadOnly(True)
        self.literature_results.setPlaceholderText("Exact retrieved passages and provenance appear here.")
        literature_layout.addWidget(self.literature_results, 3)
        self.workspace_tabs.addTab(literature_page, "Literature")
        self.literature_library_box.itemChanged.connect(self._on_literature_library_selection_changed)
        self.literature_library_box.currentRowChanged.connect(lambda _row: self._refresh_literature_documents())

        self.waveform_event_box.currentIndexChanged.connect(self._on_waveform_event_selection_changed)
        self.waveform_channel_box.currentIndexChanged.connect(self._on_waveform_view_selection_changed)
        self.waveform_max_display.valueChanged.connect(self._on_waveform_view_selection_changed)
        for checkbox in (
            self.waveform_show_individual,
            self.waveform_show_mean,
            self.waveform_show_median,
            self.waveform_show_std,
        ):
            checkbox.stateChanged.connect(self._render_waveform_view)
        self.waveform_event_box.lineEdit().returnPressed.connect(self._on_waveform_event_search)
        self.isi_group_box.currentIndexChanged.connect(self._reload_isi_view)
        self.isi_histogram_bins.valueChanged.connect(self._reload_isi_view)
        self.isi_page_start.valueChanged.connect(self._reload_isi_view)
        right_layout.addWidget(self.workspace_tabs, 3)
        right_layout.addWidget(QLabel("Research assistant"))
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider"))
        self.assistant_provider_box = QComboBox()
        self.assistant_provider_box.addItem("Local (Ollama)", "ollama")
        self.assistant_provider_box.addItem("OpenAI", "openai")
        provider_index = self.assistant_provider_box.findData(self._assistant_provider_kind)
        self.assistant_provider_box.setCurrentIndex(max(0, provider_index))
        provider_row.addWidget(self.assistant_provider_box, 1)
        right_layout.addLayout(provider_row)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Scientific profile"))
        self.assistant_profile_box = QComboBox()
        for profile in self._scientific_profiles.list_profiles():
            self.assistant_profile_box.addItem(
                f"{profile.name} · v{profile.version}", profile.profile_id
            )
        selected_profile_index = self.assistant_profile_box.findData(
            self._assistant_scientific_profile.profile_id
        )
        self.assistant_profile_box.setCurrentIndex(max(0, selected_profile_index))
        self.assistant_profile_box.currentIndexChanged.connect(self._on_assistant_profile_changed)
        profile_row.addWidget(self.assistant_profile_box, 1)
        right_layout.addLayout(profile_row)
        self.assistant_profile_details = QLabel("")
        self.assistant_profile_details.setWordWrap(True)
        right_layout.addWidget(self.assistant_profile_details)
        self.assistant_evidence_status = QLabel("Scientific evidence library: not connected")
        right_layout.addWidget(self.assistant_evidence_status)
        self._update_assistant_profile_status()
        self.ollama_host_input = QLineEdit(self._ollama_host)
        self.ollama_host_input.setPlaceholderText("http://127.0.0.1:11434")
        right_layout.addWidget(self.ollama_host_input)
        model_row = QHBoxLayout()
        self.ollama_model_box = QComboBox()
        if self._ollama_model:
            self.ollama_model_box.addItem(self._ollama_model, self._ollama_model)
        self.ollama_refresh_button = QPushButton("Refresh local models")
        self.ollama_refresh_button.clicked.connect(self.refresh_ollama_models)
        model_row.addWidget(self.ollama_model_box, 1)
        model_row.addWidget(self.ollama_refresh_button)
        right_layout.addLayout(model_row)
        self.assistant_provider_box.currentIndexChanged.connect(
            self._on_assistant_provider_changed
        )
        self.ollama_model_box.currentIndexChanged.connect(self._on_ollama_model_changed)
        self.ollama_host_input.editingFinished.connect(self._on_ollama_host_changed)
        self.assistant_status_label = QLabel("Offline")
        right_layout.addWidget(self.assistant_status_label)
        self.assistant_activity_label = QLabel("Ready")
        right_layout.addWidget(self.assistant_activity_label)
        self.chat_history = QTextEdit(); self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit(); self.chat_input.setPlaceholderText("Ask about this dataset or its analysis…")
        right_layout.addWidget(self.chat_history, 1)
        self.assistant_proposal_list = QListWidget()
        self.assistant_proposal_list.setMaximumHeight(95)
        self.assistant_proposal_list.currentItemChanged.connect(
            self._on_assistant_proposal_selected
        )
        right_layout.addWidget(self.assistant_proposal_list)
        self.assistant_proposal_details = QTextEdit()
        self.assistant_proposal_details.setReadOnly(True)
        self.assistant_proposal_details.setMaximumHeight(135)
        right_layout.addWidget(self.assistant_proposal_details)
        proposal_actions = QHBoxLayout()
        self.assistant_approve_button = QPushButton("Review & Approve")
        self.assistant_approve_button.clicked.connect(self._approve_assistant_proposal)
        self.assistant_dismiss_button = QPushButton("Dismiss")
        self.assistant_dismiss_button.clicked.connect(self._dismiss_assistant_proposal)
        proposal_actions.addWidget(self.assistant_approve_button)
        proposal_actions.addWidget(self.assistant_dismiss_button)
        right_layout.addLayout(proposal_actions)
        self.chat_input.returnPressed.connect(self.send_chat)
        chat_actions = QHBoxLayout()
        self.assistant_send_button = QPushButton("Send")
        self.assistant_send_button.clicked.connect(
            lambda _checked=False: self.send_chat()
        )
        self.assistant_cancel_button = QPushButton("Cancel request")
        self.assistant_cancel_button.clicked.connect(self.cancel_assistant_request)
        self.assistant_clear_button = QPushButton("Clear conversation")
        self.assistant_clear_button.clicked.connect(self.clear_assistant_conversation)
        self.assistant_retry_button = QPushButton("Retry")
        self.assistant_retry_button.clicked.connect(self.retry_assistant_request)
        chat_actions.addWidget(self.assistant_send_button)
        chat_actions.addWidget(self.assistant_cancel_button)
        chat_actions.addWidget(self.assistant_retry_button)
        chat_actions.addWidget(self.assistant_clear_button)
        right_layout.addWidget(self.chat_input)
        right_layout.addLayout(chat_actions)
        self.assistant_cancel_button.setEnabled(False)
        self.assistant_retry_button.setEnabled(False)
        self.assistant_approve_button.setEnabled(False)
        self.assistant_dismiss_button.setEnabled(False)
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(left_content)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([380, 900])
        layout.addWidget(splitter)
        self.setCentralWidget(root)

    def load_dataset(self):
        if any(
            thread.isRunning() and worker._view_name == "mua"
            for thread, worker in self._view_workers
        ):
            QMessageBox.warning(
                self,
                "MUA view is reading the dataset",
                "Wait for the MUA source overlay to finish before replacing the dataset.",
            )
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open biological signal dataset", "",
            "Supported data (*.npy *.mat *.csv *.txt)")
        if not filename: return
        path = Path(filename)
        delimiter = None
        if path.suffix.lower() == ".txt":
            delimiter, ok = QInputDialog.getText(self, "Text delimiter", "Enter delimiter (\\t for tab):", text="\\t")
            if not ok: return
            delimiter = "\t" if delimiter == "\\t" else delimiter
        try:
            prepared = prepare_dataset(
                path,
                delimiter=delimiter,
                select_mat_variable=self._select_mat_variable,
                confirm_channel_axis=self._confirm_channel_axis,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not load dataset", str(exc)); return
        if prepared is None:
            return
        prepared = self._confirm_raw_voltage_if_requested(prepared)
        prepared = self._resolve_project_source(prepared)
        if prepared is None:
            return
        self._commit_dataset(prepared)

    def _resolve_project_source(self, prepared: PreparedDataset) -> PreparedDataset | None:
        """Require an explicit choice before reusing a saved source identity."""
        if self.store is None:
            return prepared
        saved = self.store.load_profile()
        if saved is None:
            return prepared
        comparison = self.store.source_match(prepared.profile)
        if comparison.get("status") in {
            "partial_match_confirmation_required", "identity_unavailable"
        }:
            identity_unknown = comparison.get("status") == "identity_unavailable"
            choice = QMessageBox.question(
                self,
                "Reconnect project dataset",
                (
                    "This legacy project has no stored content fingerprint, so the source "
                    "cannot be verified automatically. Reconnect only if you confirm it is "
                    "the original recording?"
                    if identity_unknown else
                    "The candidate matches the saved structure and bounded partial fingerprint. "
                    "This is not proof that every file byte is unchanged. Reconnect it to the "
                    "saved dataset identity and metadata?"
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if choice == QMessageBox.Cancel:
                prepared.close()
                return None
            if choice == QMessageBox.Yes:
                profile = saved.model_copy(
                    update={"source_path": str(prepared.path.resolve())}
                )
                self.store.record_source_resolution(
                    "researcher_confirmed_unverified"
                    if identity_unknown else "researcher_confirmed_partial_match",
                    comparison,
                )
                exploration = DatasetExploration(prepared.array, profile)
                return PreparedDataset(prepared.path, prepared.array, profile, exploration)
        else:
            choice = QMessageBox.question(
                self,
                "Different dataset source",
                "This source does not match the project's saved dataset identity. "
                "Load it as a separate dataset? Historical artifacts will remain tied "
                "to the saved source.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                prepared.close()
                return None
        distinct_id = f"separate-{uuid4().hex}"
        profile = prepared.profile.model_copy(update={"dataset_id": distinct_id})
        exploration = DatasetExploration(prepared.array, profile)
        return PreparedDataset(prepared.path, prepared.array, profile, exploration)

    def _confirm_raw_voltage_if_requested(
        self, prepared: PreparedDataset
    ) -> PreparedDataset:
        """Record a signal-kind confirmation only after an explicit researcher choice."""
        status = (
            "confirmed"
            if "signal_kind" in prepared.profile.confirmed_fields
            else "inferred"
            if "signal_kind" in prepared.profile.inferred_fields
            else "unknown"
        )
        current = prepared.profile.metadata.signal_kind
        if status == "confirmed" or current not in (None, "raw_voltage"):
            return prepared
        choice = QMessageBox.question(
            self,
            "Confirm signal type",
            "Can you confirm this dataset is continuous raw voltage? Yes records that explicit metadata; No leaves signal type unknown and spike detection unavailable.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return prepared
        metadata = prepared.profile.metadata.model_copy(
            update={"signal_kind": "raw_voltage"}
        )
        confirmed = list(dict.fromkeys([*prepared.profile.confirmed_fields, "signal_kind"]))
        inferred = [name for name in prepared.profile.inferred_fields if name != "signal_kind"]
        unknown = [name for name in prepared.profile.unknown_fields if name != "signal_kind"]
        profile = prepared.profile.model_copy(
            update={
                "metadata": metadata,
                "confirmed_fields": confirmed,
                "inferred_fields": inferred,
                "unknown_fields": unknown,
            }
        )
        exploration = DatasetExploration(prepared.array, profile)
        return PreparedDataset(prepared.path, prepared.array, profile, exploration)

    def _select_mat_variable(self, message: str) -> str | None:
        variable_name, accepted = QInputDialog.getText(
            self,
            "Select MATLAB variable",
            f"{message}\nEnter the variable name to load:",
        )
        return variable_name if accepted else None

    def _confirm_channel_axis(self, shape: tuple[int, ...]) -> int | None:
        choice, accepted = QInputDialog.getItem(
            self,
            "Confirm array orientation",
            f"Which axis represents channels in the array with shape {shape}?",
            ["Axis 0 (rows are channels)", "Axis 1 (columns are channels)"],
            0,
            False,
        )
        if not accepted:
            return None
        return 0 if choice.startswith("Axis 0") else 1

    def _commit_dataset(self, prepared: PreparedDataset) -> None:
        """Apply a completely prepared dataset to the active view state."""
        if self._execution_in_flight or (
            self._worker_thread is not None and self._worker_thread.isRunning()
        ):
            prepared.close()
            QMessageBox.warning(
                self,
                "Analysis running",
                "Wait for the approved operation to finish before replacing the dataset.",
            )
            return
        previous = self.active_dataset
        previous_registration = self.active_registration
        summary = prepared.exploration.summary
        metadata_text = metadata_panel_text(prepared)
        navigation_state = NavigationState.from_exploration(prepared.exploration)
        registration = self.dataset_store.register(prepared.exploration)
        self._dataset_generation += 1
        self._assistant_generation += 1
        self._invalidate_prepared("dataset changed")
        if self._pipeline_prepared is not None and self._pipeline_run_state is not None:
            _pipeline, prepared_node_id, _plan, _prepared = self._pipeline_prepared
            self.pipeline_service.discard_prepared(
                self._pipeline_run_state.run_id, prepared_node_id
            )
        self._pipeline_prepared = None
        self._pipeline_run_state = None
        historical_dataset_id = (
            self._current_record.input_reference.dataset_id
            if self._current_record is not None and self._current_record.input_reference is not None
            else None
        )
        self._result_stale = (
            historical_dataset_id is not None
            and historical_dataset_id != registration.dataset.dataset_id
        )
        self._render_cache.clear()
        self._render_cache_key = None
        self._rendered_visible_window = None
        channel_items = (
            [("Signal 1", 0)]
            if summary.channel_axis is None
            else [(f"Channel {index + 1}", index) for index in range(summary.channel_count)]
        )

        self.active_dataset = prepared
        self.active_registration = registration
        if self.store is not None:
            try:
                self.store.save_profile(prepared.profile)
            except (OSError, TypeError, ValueError) as exc:
                self.operation_review.append(f"\nDataset profile could not be saved: {exc}")
        if previous is not None and previous is not prepared:
            if previous_registration is not None:
                self.dataset_store.unregister(previous_registration.session_handle)
            else:
                previous.close()
        self.navigation_state = navigation_state
        self.project_label.setText(prepared.path.name)
        self.profile_view.setPlainText(metadata_text)
        self.display_channel_list.blockSignals(True)
        self.display_channel_list.clear()
        for channel_index in range(summary.channel_count):
            item = QListWidgetItem(f"Channel {channel_index}")
            item.setData(Qt.UserRole, channel_index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.display_channel_list.addItem(item)
        self.display_channel_list.blockSignals(False)
        show_channel_controls = summary.channel_axis is not None
        self.display_channel_list.setVisible(show_channel_controls)
        self.select_all_channels_button.setVisible(show_channel_controls)
        self.clear_channels_button.setVisible(show_channel_controls)
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        for label, index in channel_items:
            self.channel_box.addItem(label, index)
        self.channel_box.setCurrentIndex(0)
        self.channel_box.blockSignals(False)
        self.mua_channel_box.blockSignals(True)
        self.mua_channel_box.clear()
        self.mua_channel_box.addItem("Select one source channel", None)
        for channel_index in range(summary.channel_count):
            self.mua_channel_box.addItem(f"Channel {channel_index}", channel_index)
        self.mua_channel_box.setCurrentIndex(0)
        self.mua_channel_box.blockSignals(False)
        self.mua_calculation_channels.blockSignals(True)
        self.mua_calculation_channels.clear()
        for channel_index in range(summary.channel_count):
            item = QListWidgetItem(f"Channel {channel_index}")
            item.setData(Qt.UserRole, channel_index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.mua_calculation_channels.addItem(item)
        self.mua_calculation_channels.blockSignals(False)
        self._redraw_navigation_view()
        self._refresh_detector_execution_choices()
        self._update_action_state()
        if self._current_record is not None and self._result_stale:
            self.analysis_state_label.setText(
                "Stale configuration — prior result retained; it belongs to a different dataset identity."
            )
        elif self._current_record is not None:
            self.analysis_state_label.setText(
                "Dataset reconnected — historical events are available; analysis was not rerun."
            )
        else:
            self.analysis_state_label.setText("Dataset ready — select a channel and prepare analysis.")
        if self._current_waveform_record is not None:
            self.waveform_display_label.setText(
                "Historical/stale waveform result retained from the previous dataset."
            )
        if self._current_isi_record is not None:
            self.isi_display_label.setText(
                "Historical/stale ISI result retained from the previous dataset."
            )
        if self._current_mua_record is not None:
            self._view_generations["mua"] += 1
            self.mua_display_label.setText(
                "Historical/stale calcMUA result retained from the previous dataset."
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.detector_box,
            self.channel_box,
            self.threshold,
            self.signed_mode_box,
            self.signed_threshold,
            self.noise_override,
            self.waveform_source_box,
            self.waveform_channel_filter,
            self.waveform_only_selected_event,
            self.waveform_pre_samples,
            self.waveform_post_samples,
            self.isi_source_box,
            self.isi_polarity_box,
            self.isi_threshold_enabled,
            self.isi_threshold,
            self.isi_threshold_units,
            self.mua_channel_box,
            self.mua_calculation_channels,
            self.mua_display_results,
            self.mua_average_inputs,
            self.mua_stdmin,
            self.mua_bandwidth,
            self.mua_mode_box,
            self.mua_num_points,
        ):
            widget.setEnabled(enabled)
        self.noise_override.setEnabled(
            enabled
            and self.detector_box.currentData() == SIGNED_PEAK_ID
            and self.signed_mode_box.currentData() == "absolute"
        )

    def _update_detector_controls(self) -> None:
        signed = self.detector_box.currentData() == SIGNED_PEAK_ID
        self.negative_controls.setVisible(not signed)
        self.signed_controls.setVisible(signed)
        absolute_mode = self.signed_mode_box.currentData() == "absolute"
        self.noise_override.setEnabled(signed and absolute_mode)
        if signed:
            try:
                value = float(self.signed_threshold.text())
                polarity = "none" if value == 0 else "positive" if value > 0 else "negative"
                label = polarity
            except ValueError:
                label = "unresolved until a finite parameter is entered"
            self.polarity_label.setText(f"Polarity: {label}")
            self.detector_explanation.setText(
                "Uncentered median absolute amplitude / 0.6745; signed threshold, "
                "SciPy peak prominence and strict amplitude cap. No baseline removal or filtering is applied."
            )
        else:
            self.detector_explanation.setText(
                "Negative-going local minima using median-centered MAD. The threshold_sd "
                "parameter is a negative noise multiplier; this convention differs from signed peaks."
            )

    def _update_action_state(self) -> None:
        has_dataset = self.active_dataset is not None and self.active_registration is not None
        running = self._execution_in_flight or (
            self._worker_thread is not None and self._worker_thread.isRunning()
        )
        self._set_controls_enabled(has_dataset and not running)
        self.load_button.setEnabled(not running)
        self.project_button.setEnabled(not running)
        self.prepare_button.setEnabled(has_dataset and not running)
        self.prepare_waveforms_button.setEnabled(
            has_dataset and self.store is not None
            and self.waveform_source_box.currentData() is not None and not running
        )
        self.prepare_isi_button.setEnabled(
            has_dataset and self.store is not None
            and self.isi_source_box.currentData() is not None and not running
        )
        self.prepare_mua_button.setEnabled(
            has_dataset
            and self.store is not None
            and (
                any(
                    self.mua_calculation_channels.item(i).checkState() == Qt.Checked
                    for i in range(self.mua_calculation_channels.count())
                )
                or self.mua_channel_box.currentData() is not None
            )
            and not running
        )
        self.prepare_mua_average_button.setEnabled(
            self.store is not None
            and sum(
                self.mua_average_inputs.item(i).checkState() == Qt.Checked
                for i in range(self.mua_average_inputs.count())
            ) >= 2
            and not running
        )
        self.approve_button.setEnabled(
            (self._prepared_execution is not None or self._prepared_mua_batch is not None)
            and self.store is not None
            and not running
        )

    def _on_detector_changed(self, _index: int) -> None:
        self._update_detector_controls()
        self._configuration_changed("detector changed")

    def _on_signed_mode_changed(self, _index: int) -> None:
        self._update_detector_controls()
        self._configuration_changed("signed threshold mode changed")

    def _on_analysis_channel_changed(self, _index: int) -> None:
        if self.active_dataset is not None:
            self._configuration_changed("analysis channel changed")

    def _on_parameters_changed(self, _value: str) -> None:
        self._update_detector_controls()
        self._configuration_changed("detector parameters changed")

    def _configuration_changed(self, reason: str) -> None:
        self._invalidate_prepared(reason)
        if self._current_record is not None:
            self._result_stale = True
            self.analysis_state_label.setText(
                f"Stale configuration — prior result retained ({reason})."
            )
        elif self.active_dataset is not None and self._prepared_execution is None:
            self.analysis_state_label.setText(
                f"Parameters changed — prepare analysis to review ({reason})."
            )
        self._update_action_state()

    def _invalidate_prepared(self, reason: str) -> None:
        if self._prepared_execution is None and self._prepared_mua_batch is None:
            return
        self._prepared_execution = None
        self._prepared_mua_batch = None
        self.operation_review.append(f"\nPrepared approval invalidated: {reason}.")
        self.analysis_state_label.setText(f"Parameters changed — prepare again ({reason}).")
        self._update_action_state()

    def _on_display_channels_changed(self, _item: QListWidgetItem) -> None:
        if self.navigation_state is None:
            return
        selected = tuple(
            self.display_channel_list.item(index).data(Qt.UserRole)
            for index in range(self.display_channel_list.count())
            if self.display_channel_list.item(index).checkState() == Qt.Checked
        )
        self.navigation_state.select_channels(selected)
        self._redraw_navigation_view()

    def _set_all_display_channels(self, selected: bool) -> None:
        self.display_channel_list.blockSignals(True)
        for index in range(self.display_channel_list.count()):
            item = self.display_channel_list.item(index)
            item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
        self.display_channel_list.blockSignals(False)
        self._on_display_channels_changed(QListWidgetItem())

    def reset_navigation_view(self) -> None:
        if self.navigation_state is None:
            return
        self.navigation_state.reset()
        self._redraw_navigation_view()

    def _redraw_navigation_view(self) -> None:
        dataset = self.active_dataset
        navigation = self.navigation_state
        if dataset is None or navigation is None:
            return
        self.statusBar().clearMessage()

        summary = dataset.exploration.summary
        x_label, y_label = plot_axis_labels(summary)
        rate = summary.metadata["sampling_rate_hz"]
        confirmed_rate = (
            float(rate.value)
            if x_label.startswith("Time") and rate.value is not None
            else None
        )
        window = navigation.visible_window
        if (
            self._rendered_visible_window is not None
            and window != self._rendered_visible_window
            and self._prepared_execution is not None
            and not self._inspecting_event
        ):
            self._configuration_changed("analysis interval changed")
        self._rendered_visible_window = window
        left = window.start / confirmed_rate if confirmed_rate else window.start
        right = window.stop / confirmed_rate if confirmed_rate else window.stop
        plot_width = self._available_plot_width()
        cache_key = (
            id(dataset),
            window.start,
            window.stop,
            plot_width,
            navigation.selected_channels,
        )
        if cache_key != self._render_cache_key:
            self._render_cache.clear()
            for channel_index in navigation.selected_channels:
                self._render_cache[channel_index] = render_exploration_signal(
                    dataset.exploration,
                    channel_index=channel_index,
                    start_sample=window.start,
                    stop_sample=window.stop,
                    pixel_width=plot_width,
                )
            self._render_cache_key = cache_key

        self._updating_plot = True
        self.figure.clear()
        self._plot_channel_by_axis = {}
        self._cursor_lines = {}
        self._event_scatters = {}
        selected_channels = navigation.selected_channels
        if not selected_channels:
            axes = [self.figure.add_subplot(111)]
            axes[0].text(
                0.5,
                0.5,
                "No channels selected. Select one or more channels to display.",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_xlim(left, right)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(y_label)
        else:
            axes = []
            count = len(selected_channels)
            for row, channel_index in enumerate(selected_channels):
                axis = self.figure.add_subplot(
                    count,
                    1,
                    row + 1,
                    sharex=axes[0] if axes else None,
                )
                rendered = self._render_cache[channel_index]
                x_values = (
                    rendered.sample_indices / confirmed_rate
                    if confirmed_rate is not None
                    else rendered.sample_indices
                )
                axis.plot(x_values, rendered.values, linewidth=0.65)
                axis.set_ylabel(y_label)
                self._set_channel_axis_title(axis, channel_index, rendered)
                axis.set_xlim(left, right)
                self._plot_channel_by_axis[axis] = channel_index
                self._cursor_lines[axis] = axis.axvline(
                    0,
                    color="tab:orange",
                    linewidth=0.8,
                    alpha=0.8,
                    visible=False,
                )
                if row == count - 1:
                    axis.set_xlabel(x_label)
                axes.append(axis)

        if axes:
            for axis in axes:
                axis.callbacks.connect("xlim_changed", self._on_plot_limits_changed)
        self._updating_plot = False
        self.navigation_toolbar.update()
        self._refresh_detection_artists()
        self.canvas.draw_idle()
        self._sync_mua_from_navigation()

    def _sync_mua_from_navigation(self) -> None:
        if not self._mua_selection_views or not self.mua_figure.axes or self.navigation_state is None:
            return
        first = self._mua_selection_views[0]
        if any(
            item["interval"] != first["interval"]
            or item["sampling_rate_hz"] != first["sampling_rate_hz"]
            or item["sampling_rate_status"] != first["sampling_rate_status"]
            for item in self._mua_selection_views[1:]
        ):
            return
        interval_start, interval_stop = first["interval"]
        visible = self.navigation_state.visible_window
        start = max(visible.start, interval_start)
        stop = min(visible.stop, interval_stop)
        if stop <= start:
            return
        rate = (
            float(first["sampling_rate_hz"])
            if first["sampling_rate_status"] is MetadataStatus.CONFIRMED
            else None
        )
        left, right = (start / rate, stop / rate) if rate else (start, stop)
        for axis in self.mua_figure.axes:
            axis.set_xlim(left, right)

    def _available_plot_width(self) -> int:
        canvas_width = self.canvas.get_width_height()[0]
        subplot_width_fraction = (
            self.figure.subplotpars.right - self.figure.subplotpars.left
        )
        return max(1, int(canvas_width * subplot_width_fraction))

    def _on_canvas_resize(self, _event) -> None:
        if self.navigation_state is not None:
            self._redraw_navigation_view()

    def _on_cursor_motion(self, event) -> None:
        channel_index = self._plot_channel_by_axis.get(event.inaxes)
        if channel_index is None or event.xdata is None or self.active_dataset is None:
            self._hide_cursor_readout()
            return
        try:
            readout = read_original_sample(
                self.active_dataset.exploration,
                channel_index=channel_index,
                axis_coordinate=float(event.xdata),
            )
        except (TypeError, ValueError, IndexError):
            self._hide_cursor_readout()
            return

        rate_entry = self.active_dataset.exploration.summary.metadata[
            "sampling_rate_hz"
        ]
        x_position = (
            readout.time_seconds
            if readout.time_seconds is not None
            else readout.sample_index
        )
        for line in self._cursor_lines.values():
            line.set_xdata([x_position, x_position])
            line.set_visible(True)

        message = (
            f"Channel {readout.channel_index} | sample {readout.sample_index} | "
            f"value {readout.value:g}"
        )
        units = self.active_dataset.exploration.summary.metadata["units"]
        if units.value is not None and units.status.value != "unknown":
            message += f" {units.value}"
        if readout.time_seconds is not None and rate_entry.status.value == "confirmed":
            message += f" | t={readout.time_seconds:.9g} s"
        self.statusBar().showMessage(message)
        self.canvas.draw_idle()

    def _hide_cursor_readout(self) -> None:
        changed = False
        for line in self._cursor_lines.values():
            if line.get_visible():
                line.set_visible(False)
                changed = True
        self.statusBar().clearMessage()
        if changed:
            self.canvas.draw_idle()

    def _on_plot_limits_changed(self, axis) -> None:
        if self._updating_plot or self.navigation_state is None:
            return
        summary = self.active_dataset.exploration.summary
        rate = summary.metadata["sampling_rate_hz"]
        sampling_rate = (
            float(rate.value)
            if rate.status.value == "confirmed" and rate.value is not None
            else None
        )
        try:
            self.navigation_state.set_from_axis_limits(
                *axis.get_xlim(), sampling_rate_hz=sampling_rate
            )
        except ValueError:
            return
        window = self.navigation_state.visible_window
        left = window.start / sampling_rate if sampling_rate else window.start
        right = window.stop / sampling_rate if sampling_rate else window.stop
        self._updating_plot = True
        for current_axis in self.figure.axes:
            current_axis.set_xlim(left, right)
        self._updating_plot = False
        if not self._inspecting_event and self._prepared_execution is not None:
            self._configuration_changed("analysis interval changed")
        self._update_rendered_lines()

    def _update_rendered_lines(self) -> None:
        """Refresh existing plot artists after zoom or pan without rebuilding axes."""
        dataset = self.active_dataset
        navigation = self.navigation_state
        if dataset is None or navigation is None or not self._plot_channel_by_axis:
            return
        window = navigation.visible_window
        plot_width = self._available_plot_width()
        cache_key = (
            id(dataset),
            window.start,
            window.stop,
            plot_width,
            navigation.selected_channels,
        )
        if cache_key != self._render_cache_key:
            self._render_cache.clear()
            for channel_index in navigation.selected_channels:
                self._render_cache[channel_index] = render_exploration_signal(
                    dataset.exploration,
                    channel_index=channel_index,
                    start_sample=window.start,
                    stop_sample=window.stop,
                    pixel_width=plot_width,
                )
            self._render_cache_key = cache_key

        rate = dataset.exploration.summary.metadata["sampling_rate_hz"]
        confirmed_rate = (
            float(rate.value)
            if rate.status.value == "confirmed" and rate.value is not None
            else None
        )
        left = window.start / confirmed_rate if confirmed_rate else window.start
        right = window.stop / confirmed_rate if confirmed_rate else window.stop
        self._updating_plot = True
        for axis, channel_index in self._plot_channel_by_axis.items():
            rendered = self._render_cache[channel_index]
            x_values = (
                rendered.sample_indices / confirmed_rate
                if confirmed_rate is not None
                else rendered.sample_indices
            )
            axis.lines[0].set_data(x_values, rendered.values)
            self._set_channel_axis_title(axis, channel_index, rendered)
            axis.set_xlim(left, right)
        self._updating_plot = False
        self._refresh_detection_artists()
        self.canvas.draw_idle()

    @staticmethod
    def _set_channel_axis_title(axis, channel_index: int, rendered: RenderedSignal) -> None:
        title = f"Channel {channel_index} (original index)"
        if rendered.reduced:
            title += " — min/max envelope (display only)"
        axis.set_title(title)

    def _pipeline_from_editor(self) -> PipelineDefinition:
        if not self.pipeline_json_editor.toPlainText().strip():
            raise ValueError("Create or open a pipeline before continuing.")
        return PipelineDefinition.model_validate_json(
            self.pipeline_json_editor.toPlainText(), strict=True
        )

    def _pipeline_history_records(self) -> dict[str, ExecutionRecord]:
        records = {record.execution_id: record for record in self._execution_records}
        if self.store is not None:
            restored, _rejected = self.store.restore_execution_records(self.operation_registry)
            records.update({record.execution_id: record for record in restored})
        return records

    def _pipeline_available_artifacts(self) -> set[str]:
        if self.store is None:
            return set()
        return {
            f"artifact://{item['kind']}/{item['artifact_id']}"
            for item in self.store.discover_artifacts()
            if item.get("status") == "available"
        }

    def _pipeline_node_records(self, pipeline, records, state=None):
        result = {}
        for node in pipeline.nodes:
            if node.reuse_execution_id and node.reuse_execution_id in records:
                result[node.node_id] = records[node.reuse_execution_id]
        if state is not None:
            for node_id, execution_id in state.execution_ids.items():
                if execution_id in records:
                    result[node_id] = records[execution_id]
        return result

    def _new_pipeline_draft(self) -> None:
        name, accepted = QInputDialog.getText(self, "New pipeline", "Pipeline name:")
        if not accepted or not name.strip():
            return
        pipeline = new_pipeline(name.strip(), ())
        self.pipeline_json_editor.setPlainText(pipeline.model_dump_json(indent=2))
        self._pipeline_run_state = None
        self._pipeline_prepared = None
        self.pipeline_status.setPlainText(
            "Draft created. Add nodes as JSON using registered operation IDs, explicit parameters, and typed bindings."
        )

    def _pipeline_add_operation_node(self) -> None:
        try:
            payload = json.loads(self.pipeline_json_editor.toPlainText())
            pipeline = PipelineDefinition.model_validate_json(
                self.pipeline_json_editor.toPlainText(), strict=True
            )
            operations = self.operation_registry.list_operations()
            labels = [f"{item.operation_id} v{item.version} — {item.name}" for item in operations]
            selected, accepted = QInputDialog.getItem(
                self, "Add registered operation", "Operation:", labels, 0, False
            )
            if not accepted:
                return
            operation = operations[labels.index(selected)]
            stem = operation.operation_id.split(".")[-1].replace("-", "_")
            existing_ids = {node.node_id for node in pipeline.nodes}
            suffix = 1
            node_id = stem
            while node_id in existing_ids:
                suffix += 1
                node_id = f"{stem}_{suffix}"
            payload["nodes"].append({
                "node_id": node_id,
                "operation_id": operation.operation_id,
                "operation_version": operation.version,
                "parameters": {},
                "inputs": [],
                "depends_on": [],
                "reuse_execution_id": None,
            })
            self.pipeline_json_editor.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.pipeline_status.setPlainText(
                f"Added {operation.operation_id} v{operation.version} as node {node_id}. "
                "No parameter values were filled; configure them explicitly before validation."
            )
        except Exception as exc:  # noqa: BLE001 - pipeline JSON is researcher-authored input.
            QMessageBox.warning(self, "Could not add operation", str(exc))

    def _pipeline_add_dataset_binding(self) -> None:
        if self.active_registration is None:
            QMessageBox.warning(self, "No active dataset", "Load a dataset before adding a dataset binding.")
            return
        try:
            pipeline = self._pipeline_from_editor()
            node_candidates = []
            for node in pipeline.nodes:
                definition = self.operation_registry.get(node.operation_id, version=node.operation_version)
                field = definition.input_model.model_fields.get("signal")
                if field is not None and getattr(field.annotation, "__name__", "") == "DataReference":
                    node_candidates.append(node.node_id)
            if not node_candidates:
                raise ValueError("No node in this pipeline declares a DataReference signal input.")
            node_id, accepted = QInputDialog.getItem(
                self, "Bind active dataset", "Signal input node:", node_candidates, 0, False
            )
            if not accepted:
                return
            registration = self.active_registration
            channel, accepted = QInputDialog.getInt(
                self, "Select channel", "Zero-based channel index:", 0,
                0, registration.channel_count - 1, 1,
            )
            if not accepted:
                return
            start, accepted = QInputDialog.getInt(
                self, "Sample interval", "Start sample (inclusive):", 0,
                0, registration.sample_count - 1, 1,
            )
            if not accepted:
                return
            stop, accepted = QInputDialog.getInt(
                self, "Sample interval", "Stop sample (exclusive):", registration.sample_count,
                start + 1, registration.sample_count, 1,
            )
            if not accepted:
                return
            binding = {
                "target_field": "signal",
                "source_kind": "dataset",
                "dataset": registration.dataset.model_dump(mode="json"),
                "channels": [channel],
                "sample_interval": {"start": start, "stop": stop},
                "component_id": None,
                "source_node_id": None,
                "source_port": None,
                "artifact_value": None,
            }
            payload = json.loads(self.pipeline_json_editor.toPlainText())
            target = next(item for item in payload["nodes"] if item["node_id"] == node_id)
            target["inputs"] = [item for item in target.get("inputs", []) if item.get("target_field") != "signal"]
            target.setdefault("inputs", []).append(binding)
            self.pipeline_json_editor.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.pipeline_status.setPlainText(
                f"Bound the active dataset identity to node {node_id}, channel {channel}, "
                f"samples [{start}, {stop}). No session handle was saved."
            )
        except Exception as exc:  # noqa: BLE001 - report invalid node/source selections.
            QMessageBox.warning(self, "Could not bind dataset", str(exc))

    def _pipeline_connect_nodes(self) -> None:
        try:
            pipeline = self._pipeline_from_editor()
            detector_ids = {NEGATIVE_DETECTOR_ID, SIGNED_PEAK_ID}
            candidates = []
            for source in pipeline.nodes:
                for target in pipeline.nodes:
                    if source.node_id == target.node_id:
                        continue
                    if source.operation_id in detector_ids and target.operation_id == "neuroscope.spike.extract-waveforms":
                        target_field, source_port = "detections", "detection_events"
                    elif source.operation_id in detector_ids and target.operation_id == "neuroscope.spike.analyze-isi":
                        target_field, source_port = "events", "detection_events"
                    elif source.operation_id == CALC_MUA_ID and target.operation_id == AVERAGE_MUA_OPERATION.operation_id:
                        target_field, source_port = "traces", "mua_trace"
                    else:
                        continue
                    candidates.append((source, target, target_field, source_port))
            if not candidates:
                raise ValueError("No compatible source/destination operation pair is present.")
            labels = [f"{source.node_id} → {target.node_id} ({field})" for source, target, field, _port in candidates]
            selected, accepted = QInputDialog.getItem(
                self, "Connect compatible nodes", "Typed operation edge:", labels, 0, False
            )
            if not accepted:
                return
            source, target, target_field, source_port = candidates[labels.index(selected)]
            payload = json.loads(self.pipeline_json_editor.toPlainText())
            target_data = next(item for item in payload["nodes"] if item["node_id"] == target.node_id)
            edge = {
                "target_field": target_field,
                "source_kind": "node_output",
                "dataset": None,
                "channels": [],
                "sample_interval": None,
                "component_id": None,
                "source_node_id": source.node_id,
                "source_port": source_port,
                "artifact_value": None,
            }
            if not any(item.get("source_node_id") == source.node_id and item.get("target_field") == target_field for item in target_data.get("inputs", [])):
                target_data.setdefault("inputs", []).append(edge)
            target_data["depends_on"] = sorted(set(target_data.get("depends_on", [])) | {source.node_id})
            self.pipeline_json_editor.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.pipeline_status.setPlainText(
                f"Connected {source.node_id} → {target.node_id} using {source_port}. "
                "This is a typed handoff, not an implicit analysis transformation."
            )
        except Exception as exc:  # noqa: BLE001 - report incompatible graph edits.
            QMessageBox.warning(self, "Could not connect nodes", str(exc))

    def _pipeline_select_mua_artifact(self) -> None:
        if self.store is None:
            QMessageBox.warning(self, "No project", "Open a project to select saved MUA artifacts.")
            return
        try:
            pipeline = self._pipeline_from_editor()
            targets = [node.node_id for node in pipeline.nodes if node.operation_id == AVERAGE_MUA_OPERATION.operation_id]
            if not targets:
                raise ValueError("Add a registered MUA averaging node first.")
            target_id, accepted = QInputDialog.getItem(
                self, "Select target node", "MUA averaging node:", targets, 0, False
            )
            if not accepted:
                return
            available = self._pipeline_available_artifacts()
            candidates = []
            for snapshot in self.store.load_execution_history():
                if snapshot.get("operation_id") != CALC_MUA_ID or snapshot.get("status") != "succeeded":
                    continue
                try:
                    reference = mua_artifact_reference_from_snapshot(snapshot)
                except (KeyError, TypeError, ValueError):
                    continue
                if reference.artifact_reference in available:
                    label = (
                        f"Channel {reference.channel_index} — execution {reference.execution_id} — "
                        f"{reference.artifact_reference}"
                    )
                    candidates.append((label, reference))
            if len(candidates) < 2:
                raise ValueError("At least two available, lineage-verified calcMUA artifacts are required for averaging.")
            labels = [item[0] for item in candidates]
            selected, accepted = QInputDialog.getItem(
                self, "Select saved MUA artifact", "Artifact input:", labels, 0, False
            )
            if not accepted:
                return
            reference = candidates[labels.index(selected)][1]
            payload = json.loads(self.pipeline_json_editor.toPlainText())
            target = next(item for item in payload["nodes"] if item["node_id"] == target_id)
            existing = {
                item.get("artifact_value", {}).get("artifact_reference")
                for item in target.get("inputs", [])
                if item.get("source_kind") == "artifact" and isinstance(item.get("artifact_value"), dict)
            }
            if reference.artifact_reference in existing:
                raise ValueError("That artifact is already selected for this averaging node.")
            target.setdefault("inputs", []).append({
                "target_field": "traces",
                "source_kind": "artifact",
                "dataset": None,
                "channels": [],
                "sample_interval": None,
                "component_id": None,
                "source_node_id": None,
                "source_port": None,
                "artifact_value": reference.model_dump(mode="json"),
            })
            self.pipeline_json_editor.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.pipeline_status.setPlainText(
                f"Selected saved calcMUA artifact {reference.artifact_reference} from execution "
                f"{reference.execution_id}. The averaging node will validate scientific compatibility before approval."
            )
        except Exception as exc:  # noqa: BLE001 - report artifact availability and compatibility issues.
            QMessageBox.warning(self, "Could not select artifact", str(exc))

    def _pipeline_reuse_execution(self) -> None:
        if self.store is None:
            QMessageBox.warning(self, "No project", "Open a project to select a saved execution.")
            return
        try:
            payload = json.loads(self.pipeline_json_editor.toPlainText())
            pipeline = self._pipeline_from_editor()
            candidates = []
            for record in self._pipeline_history_records().values():
                if record.status is not ExecutionStatus.SUCCEEDED:
                    continue
                try:
                    definition = self.operation_registry.get(record.operation_id, version=record.operation_version)
                except (KeyError, ValueError):
                    continue
                if definition.implementation_version != record.implementation_version:
                    continue
                candidates.append(record)
            if not candidates:
                raise ValueError("No successful execution with a currently supported implementation is available.")
            labels = [
                f"{record.operation_id} v{record.operation_version} — {record.execution_id}"
                for record in candidates
            ]
            selected, accepted = QInputDialog.getItem(
                self, "Reuse saved execution", "Historical result (no rerun):", labels, 0, False
            )
            if not accepted:
                return
            record = candidates[labels.index(selected)]
            stem = f"reuse_{record.operation_id.split('.')[-1].replace('-', '_')}"
            existing_ids = {node.node_id for node in pipeline.nodes}
            node_id = stem
            suffix = 1
            while node_id in existing_ids:
                suffix += 1
                node_id = f"{stem}_{suffix}"
            payload["nodes"].append({
                "node_id": node_id,
                "operation_id": record.operation_id,
                "operation_version": record.operation_version,
                "parameters": {},
                "inputs": [],
                "depends_on": [],
                "reuse_execution_id": record.execution_id,
            })
            self.pipeline_json_editor.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
            self.pipeline_status.setPlainText(
                f"Selected existing execution {record.execution_id} as reused node {node_id}. "
                "Its original execution identity is preserved; it will not be rerun or reapproved."
            )
        except Exception as exc:  # noqa: BLE001 - report project-history selection errors.
            QMessageBox.warning(self, "Could not reuse execution", str(exc))

    def _open_saved_pipeline(self) -> None:
        if self.store is None:
            QMessageBox.warning(self, "No project", "Select or open a project first.")
            return
        entries = self.store.list_pipelines()
        available = [item["pipeline"] for item in entries if item.get("status") == "available"]
        if not available:
            QMessageBox.information(self, "No saved pipelines", "This project has no supported pipeline definitions.")
            return
        labels = [f"{item.name} — r{item.revision} ({item.pipeline_id})" for item in available]
        selected, accepted = QInputDialog.getItem(
            self, "Open pipeline", "Select a saved pipeline:", labels, 0, False
        )
        if not accepted:
            return
        pipeline = available[labels.index(selected)]
        self.pipeline_json_editor.setPlainText(pipeline.model_dump_json(indent=2))
        self._pipeline_run_state = None
        self._pipeline_prepared = None
        saved_runs = [
            item["run"] for item in self.store.list_pipeline_runs()
            if item.get("status") == "available"
            and item["run"].pipeline_id == pipeline.pipeline_id
            and item["run"].pipeline_revision == pipeline.revision
        ]
        run_text = "\n".join(
            f"Run {item.run_id}: " + ", ".join(
                f"{node}={status}" for node, status in item.node_status.items()
            )
            for item in saved_runs[-5:]
        ) or "No saved run snapshots for this revision."
        self.pipeline_status.setPlainText(
            f"Opened {pipeline.name!r} revision {pipeline.revision}. Opening did not execute or approve any node.\n\n"
            f"Saved run snapshots (inspection only):\n{run_text}"
        )

    def _validate_pipeline_ui(self) -> None:
        try:
            pipeline = self._pipeline_from_editor()
            records = self._pipeline_history_records()
            completed = (
                {node for node, status in self._pipeline_run_state.node_status.items() if status in {"succeeded", "reused"}}
                if self._pipeline_run_state is not None
                else set()
            )
            report = validate_pipeline(
                pipeline,
                self.operation_registry,
                dataset_store=self.dataset_store,
                execution_records=records,
                completed_node_ids=completed,
                available_artifact_references=self._pipeline_available_artifacts(),
                node_records=self._pipeline_node_records(pipeline, records, self._pipeline_run_state),
            )
            lines = [
                f"Pipeline {pipeline.name!r}, revision {pipeline.revision}",
                f"Structurally valid: {report.structurally_valid}",
                f"At least one node ready: {report.execution_ready}",
                f"Deterministic dependency order: {', '.join(report.topological_order) or '(empty)'}",
            ]
            for node in report.nodes:
                lines.append(f"\n{node.node_id}: {node.status}")
                lines.extend(f"  Blocked/invalid: {reason}" for reason in node.reasons)
                lines.extend(f"  Warning: {warning}" for warning in node.warnings)
            lines.extend(f"\nError: {error}" for error in report.errors)
            self.pipeline_status.setPlainText("\n".join(lines))
        except Exception as exc:  # noqa: BLE001 - user-authored definitions are untrusted input.
            self.pipeline_status.setPlainText(f"Pipeline validation failed: {exc}")

    def _save_pipeline_ui(self) -> None:
        if self.store is None:
            QMessageBox.warning(self, "No project", "Select or open a project before saving a pipeline.")
            return
        try:
            pipeline = self._pipeline_from_editor()
            saved = pipeline
            try:
                existing = self.store.load_pipeline(pipeline.pipeline_id)
            except FileNotFoundError:
                existing = None
            if existing is not None and pipeline.revision <= existing.revision:
                from ..analysis.pipelines import revise_pipeline

                saved = revise_pipeline(
                    existing,
                    name=pipeline.name,
                    description=pipeline.description,
                    nodes=pipeline.nodes,
                )
            self.store.save_pipeline(saved)
            self.pipeline_json_editor.setPlainText(saved.model_dump_json(indent=2))
            self.pipeline_status.setPlainText(
                f"Saved pipeline revision {saved.revision}. Previous revisions remain unchanged."
            )
            self._pipeline_run_state = None
            self._pipeline_prepared = None
        except Exception as exc:  # noqa: BLE001 - validation and file errors are shown to researcher.
            QMessageBox.critical(self, "Pipeline save failed", str(exc))

    def _start_pipeline_ui(self) -> None:
        if self._execution_in_flight:
            QMessageBox.warning(self, "Execution running", "Wait for the active node before starting another run.")
            return
        if self._pipeline_prepared is not None:
            QMessageBox.warning(self, "Prepared node pending", "Approve or dismiss the reviewed node before starting another run.")
            return
        if self.store is None:
            QMessageBox.warning(self, "No project", "Select or open a project before starting a pipeline.")
            return
        try:
            pipeline = self._pipeline_from_editor()
            saved = self.store.load_pipeline(pipeline.pipeline_id, pipeline.revision)
            if saved != pipeline:
                raise ValueError("The editor definition differs from the saved pipeline revision; save a new revision first.")
            records = self._pipeline_history_records()
            validation = validate_pipeline(
                pipeline, self.operation_registry, dataset_store=self.dataset_store,
                execution_records=records,
                available_artifact_references=self._pipeline_available_artifacts(),
                node_records=self._pipeline_node_records(pipeline, records),
            )
            if not validation.structurally_valid:
                raise ValueError("Pipeline structure is invalid. Validate and fix node definitions first.")
            state = self.pipeline_service.start(pipeline)
            state = self.pipeline_service.bind_reused_nodes(
                pipeline, state, records,
                available_artifact_references=self._pipeline_available_artifacts(),
            )
            self.store.save_pipeline_run(state)
            self._pipeline_run_state = state
            self._pipeline_prepared = None
            self.pipeline_status.setPlainText(
                f"Started pipeline run {state.run_id} for revision {state.pipeline_revision}. "
                "No scientific node has executed. Prepare and approve each node individually."
            )
        except Exception as exc:  # noqa: BLE001 - display project/validation blockers.
            QMessageBox.warning(self, "Pipeline cannot start", str(exc))

    def _prepare_pipeline_node_ui(self) -> None:
        if self._execution_in_flight:
            QMessageBox.warning(self, "Execution running", "Wait for the active pipeline node to finish.")
            return
        if self._pipeline_prepared is not None:
            QMessageBox.warning(self, "Node already prepared", "Approve or dismiss the currently reviewed node first.")
            return
        if self.store is None or self._pipeline_run_state is None:
            QMessageBox.warning(self, "No pipeline run", "Start a saved pipeline run first.")
            return
        try:
            pipeline = self._pipeline_from_editor()
            records = self._pipeline_history_records()
            completed = {
                node for node, status in self._pipeline_run_state.node_status.items()
                if status in {"succeeded", "reused"}
            }
            report = validate_pipeline(
                pipeline, self.operation_registry, dataset_store=self.dataset_store,
                execution_records=records, completed_node_ids=completed,
                available_artifact_references=self._pipeline_available_artifacts(),
                node_records=self._pipeline_node_records(
                    pipeline, records, self._pipeline_run_state
                ),
            )
            if not report.structurally_valid:
                raise ValueError("Pipeline structure is invalid; no node was prepared.")
            validation_by_id = {item.node_id: item for item in report.nodes}
            next_node = None
            for node_id in report.topological_order:
                if self._pipeline_run_state.node_status[node_id] != "pending":
                    continue
                if validation_by_id[node_id].status == "ready":
                    next_node = node_id
                    break
            if next_node is None:
                raise ValueError("No pipeline node is currently ready. Review blocked dependencies and metadata.")
            node_records = {
                node_id: records[execution_id]
                for node_id, execution_id in self._pipeline_run_state.execution_ids.items()
                if execution_id in records
            }
            plan, prepared = self.pipeline_service.prepare_node(
                pipeline, self._pipeline_run_state, next_node,
                node_records=node_records, historical_records=records,
            )
            state_payload = self._pipeline_run_state.model_dump(mode="python")
            statuses = dict(self._pipeline_run_state.node_status)
            statuses[next_node] = "prepared"
            state_payload.update(node_status=statuses, updated_at_utc=datetime.now(timezone.utc))
            self._pipeline_run_state = PipelineRunState.model_validate(state_payload)
            self.store.save_pipeline_run(self._pipeline_run_state)
            self._pipeline_prepared = (pipeline, next_node, plan, prepared)
            self.pipeline_status.setPlainText(
                f"Prepared node {next_node}: {plan.operation_id} v{plan.operation_version}\n"
                f"Implementation: {plan.implementation_version}\n"
                f"Warnings: {json.dumps(plan.warnings, ensure_ascii=False)}\n\n"
                f"Exact request for review:\n{json.dumps(plan.request_snapshot, indent=2, ensure_ascii=False)}\n\n"
                "Preparation did not run the operation. Use the separate approval button only after reviewing this request."
            )
        except Exception as exc:  # noqa: BLE001 - show exact validation blockers.
            self.pipeline_status.setPlainText(f"Pipeline node could not be prepared: {exc}")

    def _dismiss_pipeline_node_ui(self) -> None:
        if self._execution_in_flight:
            QMessageBox.warning(self, "Execution running", "An approved node cannot be dismissed while it executes.")
            return
        if self._pipeline_prepared is None or self._pipeline_run_state is None:
            return
        _pipeline, node_id, _plan, _prepared = self._pipeline_prepared
        self.pipeline_service.discard_prepared(self._pipeline_run_state.run_id, node_id)
        statuses = dict(self._pipeline_run_state.node_status)
        statuses[node_id] = "pending"
        payload = self._pipeline_run_state.model_dump(mode="python")
        payload.update(node_status=statuses, updated_at_utc=datetime.now(timezone.utc))
        self._pipeline_run_state = PipelineRunState.model_validate(payload)
        if self.store is not None:
            self.store.save_pipeline_run(self._pipeline_run_state)
        self._pipeline_prepared = None
        self.pipeline_status.append("Prepared node dismissed. No approval or execution was recorded.")

    def _approve_pipeline_node_ui(self) -> None:
        if self._pipeline_prepared is None or self._pipeline_run_state is None or self.store is None:
            QMessageBox.warning(self, "No prepared node", "Prepare and review a node before approving it.")
            return
        if self._execution_in_flight or (self._worker_thread is not None and self._worker_thread.isRunning()):
            QMessageBox.warning(self, "Execution running", "Wait for the active scientific operation to finish.")
            return
        pipeline, node_id, plan, _prepared = self._pipeline_prepared
        try:
            if self._pipeline_from_editor() != pipeline:
                self.pipeline_service.discard_prepared(
                    self._pipeline_run_state.run_id, node_id
                )
                statuses = dict(self._pipeline_run_state.node_status)
                statuses[node_id] = "pending"
                payload = self._pipeline_run_state.model_dump(mode="python")
                payload.update(node_status=statuses, updated_at_utc=datetime.now(timezone.utc))
                self._pipeline_run_state = PipelineRunState.model_validate(payload)
                self.store.save_pipeline_run(self._pipeline_run_state)
                self._pipeline_prepared = None
                raise ValueError("Pipeline definition changed after review; validate and prepare the changed definition again.")
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Pipeline proposal is stale", str(exc))
            return
        answer = QMessageBox.question(
            self,
            "Explicitly approve pipeline node",
            f"Execute this exact node?\n\nOperation: {plan.operation_id} v{plan.operation_version}\n"
            f"Implementation: {plan.implementation_version}\nNode: {node_id}\n"
            f"Warnings: {'; '.join(plan.warnings) or 'None'}\n\n"
            "This approves only this one node. Later nodes require separate review and approval.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            approved = self.pipeline_service.approve_node(
                self._pipeline_run_state.run_id, node_id, approved_by="Researcher"
            )
            thread = QThread(self)
            worker = _PipelineNodeWorker(
                self.pipeline_service, pipeline, self._pipeline_run_state,
                node_id, approved, self.store,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._on_pipeline_node_finished)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_pipeline_worker_thread_finished)
            self._pipeline_worker_thread = thread
            self._pipeline_worker = worker
            self._execution_in_flight = True
            self.pipeline_status.append(f"Approved one node ({node_id}); executing in a worker thread…")
            thread.start()
        except Exception as exc:  # noqa: BLE001 - display approval binding failures.
            QMessageBox.critical(self, "Pipeline approval failed", str(exc))

    @Slot(object, object, object)
    def _on_pipeline_node_finished(self, state, record, error) -> None:
        self._execution_in_flight = False
        self._pipeline_prepared = None
        if error is not None:
            self.pipeline_status.append(f"Pipeline node execution failed before a record was returned: {error}")
            return
        self._pipeline_run_state = state
        if isinstance(record, ExecutionRecord):
            self._execution_records.append(record)
            status = record.status.value
            details = (
                f"; {record.error.error_type}: {record.error.message}"
                if record.error is not None else ""
            )
            self.pipeline_status.append(
                f"Node {record.operation_id}: {status}; execution {record.execution_id}{details}. "
                "Successful upstream artifacts remain available even if a later node fails."
            )
        self.pipeline_status.append(
            "Run state: " + ", ".join(f"{key}={value}" for key, value in state.node_status.items())
        )

    @Slot()
    def _on_pipeline_worker_thread_finished(self) -> None:
        self._pipeline_worker_thread = None
        self._pipeline_worker = None

    def select_project(self) -> None:
        """Create a new project, or route an existing folder through safe open."""
        if self._literature_import_thread is not None and self._literature_import_thread.isRunning():
            QMessageBox.warning(self, "Library import running", "Wait for literature extraction to finish before changing projects.")
            return
        choice = QMessageBox.question(
            self,
            "Project storage",
            "Create a new project folder? Choose No to select an existing project folder.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if choice == QMessageBox.Cancel:
            return
        try:
            if choice == QMessageBox.Yes:
                parent = QFileDialog.getExistingDirectory(
                    self, "Choose parent folder for the new project", ""
                )
                if not parent:
                    return
                name, accepted = QInputDialog.getText(
                    self, "Create project", "New project folder name:"
                )
                if not accepted:
                    return
                name = name.strip()
                if not name or Path(name).name != name or name in {".", ".."}:
                    QMessageBox.warning(self, "Invalid project name", "Enter a simple folder name.")
                    return
                project_path = Path(parent) / name
                if project_path.exists():
                    QMessageBox.warning(
                        self,
                        "Project already exists",
                        "The selected project path already exists. Select it as an existing project or choose another name.",
                    )
                    return
                project_path.mkdir(parents=False, exist_ok=False)
            else:
                selected = QFileDialog.getExistingDirectory(
                    self, "Select existing project folder", ""
                )
                if not selected:
                    return
                self.open_project(selected)
                return
            project_path = Path(project_path)
            candidate_store = ProjectStore(project_path)
            self.store = candidate_store
            self._literature_store = candidate_store.literature
            self._refresh_literature_libraries()
            self._assistant_generation += 1
            self._assistant_tools = AssistantToolService(self._assistant_state)
            self.operation_runner = OperationRunner(
                self.operation_registry,
                data_store=self.dataset_store,
                artifact_store=self.store.artifacts,
                isi_artifact_store=self.store.isi_artifacts,
                mua_artifact_store=self.store.mua_artifacts,
            )
            self.pipeline_service = PipelineExecutionService(
                self.operation_registry, self.operation_runner, self.dataset_store
            )
            self._pipeline_run_state = None
            self._pipeline_prepared = None
            if self.active_dataset is not None:
                self.store.save_profile(self.active_dataset.profile)
            self._refresh_mua_project_results()
            self._invalidate_prepared("project storage changed; prepare again")
            self.project_path_label.setText(str(project_path))
            self.analysis_state_label.setText(
                "Project ready — prepare analysis."
                if self.active_dataset is not None
                else "No dataset — project selected."
            )
            self._update_action_state()
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Project unavailable", str(exc))

    def open_project(self, project_path: str | None = None) -> None:
        """Preflight an existing project, then restore history without rerunning work."""
        if self._literature_import_thread is not None and self._literature_import_thread.isRunning():
            QMessageBox.warning(self, "Library import running", "Wait for literature extraction to finish before opening another project.")
            return
        if self._execution_in_flight or (
            self._worker_thread is not None and self._worker_thread.isRunning()
        ) or any(thread.isRunning() for thread, _worker in self._view_workers):
            QMessageBox.warning(
                self,
                "Analysis or artifact read running",
                "Wait for the active execution or artifact read before opening another project.",
            )
            return
        if project_path is None:
            project_path = QFileDialog.getExistingDirectory(
                self, "Open NeuroScope project", ""
            )
            if not project_path:
                return
        try:
            candidate = ProjectStore.open_existing(project_path)
            records, rejected = candidate.restore_execution_records(self.operation_registry)
            artifacts = candidate.discover_artifacts()
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Project could not be opened", str(exc))
            return

        # Only after the full preflight succeeds do we replace the active project.
        if self.active_registration is not None:
            try:
                self.dataset_store.unregister(self.active_registration.session_handle)
            except DatasetAccessError:
                pass
        elif self.active_dataset is not None:
            self.active_dataset.close()
        self.active_dataset = None
        self.active_registration = None
        self.navigation_state = None
        self._dataset_generation += 1
        for name in self._view_generations:
            self._view_generations[name] += 1
        self._assistant_generation += 1
        self._invalidate_prepared("project changed; prepare again")
        self.clear_assistant_conversation()
        self.store = candidate
        self._literature_store = candidate.literature
        self._assistant_external_document_allowlist = ()
        self._refresh_literature_libraries()
        self.operation_runner = OperationRunner(
            self.operation_registry,
            data_store=self.dataset_store,
            artifact_store=candidate.artifacts,
            isi_artifact_store=candidate.isi_artifacts,
            mua_artifact_store=candidate.mua_artifacts,
        )
        self.pipeline_service = PipelineExecutionService(
            self.operation_registry, self.operation_runner, self.dataset_store
        )
        self._pipeline_run_state = None
        self._pipeline_prepared = None
        self._assistant_tools = AssistantToolService(self._assistant_state)
        self._execution_records = records
        self._detector_records = {
            record.execution_id: record
            for record in records
            if record.operation_id in {NEGATIVE_DETECTOR_ID, SIGNED_PEAK_ID}
        }
        self._current_record = max(
            self._detector_records.values(),
            key=lambda item: item.finished_at_utc,
            default=None,
        )
        self._current_events = (
            self._events_from_record(self._current_record)
            if self._current_record is not None else []
        )
        self._result_stale = bool(self._current_record)
        self._current_waveform_record = None
        self._current_isi_record = None
        self._current_mua_record = None
        self._waveform_artifact_view = None
        self._isi_artifact_view = None
        self._mua_artifact_view = None
        self._mua_selection_views = []
        self._populate_saved_artifact_choices(records, artifacts)
        self._refresh_detector_execution_choices()
        self._refresh_mua_project_results()
        profile = candidate.load_profile()
        self.project_path_label.setText(str(candidate.root))
        self.project_label.setText(
            "No dataset loaded" if profile is None else f"Source offline · {profile.name}"
        )
        self.profile_view.setPlainText(
            "No saved dataset profile." if profile is None else
            "Saved dataset profile (historical; source not loaded):\n"
            + f"Project identity: {candidate.project_id or 'legacy project without manifest'}\n"
            + f"Source identity record: {candidate.source_identity_status}; any bounded fingerprint requires confirmation.\n"
            + profile.model_dump_json(indent=2)
        )
        self.figure.clear()
        self.canvas.draw_idle()
        self._populate_event_table()
        self.analysis_state_label.setText(
            f"Project recovered: {len(records)} compatible historical execution(s); "
            f"{len(rejected)} unsupported record(s); no analyses were rerun."
        )
        if (
            rejected
            or candidate.source_identity_status == "invalid"
            or any(item.get("status") != "available" for item in artifacts)
        ):
            summary = [f"Historical records unavailable: {len(rejected)}"]
            if candidate.source_identity_status == "invalid":
                summary.append("Saved source identity is malformed; reconnection requires manual confirmation.")
            summary.extend(
                f"Execution {item['execution_id']}: {item['reason']}"
                for item in rejected
            )
            summary.extend(
                f"{item.get('kind')} {item.get('artifact_id')}: {item.get('status')}"
                for item in artifacts if item.get("status") != "available"
            )
            self.operation_review.setPlainText("\n".join(summary))
        self._update_action_state()

    def _populate_saved_artifact_choices(self, records, artifacts) -> None:
        available = {
            (str(item.get("kind")), str(item.get("artifact_id")))
            for item in artifacts if item.get("status") == "available"
        }
        self.waveform_saved_box.clear()
        self.waveform_saved_box.addItem("Select saved waveform artifact", None)
        self.isi_saved_box.clear()
        self.isi_saved_box.addItem("Select saved ISI artifact", None)
        for record in records:
            output = record.output
            if record.status is not ExecutionStatus.SUCCEEDED or output is None:
                continue
            if isinstance(output, WaveformExtractionOutput):
                artifact_id = output.artifact_reference.rsplit("/", 1)[-1]
                if ("waveforms", artifact_id) in available:
                    self.waveform_saved_box.addItem(
                        f"{record.execution_id[:8]} · {record.operation_id}", record.execution_id
                    )
            elif isinstance(output, ISIOutput):
                artifact_id = output.interval_artifact_reference.rsplit("/", 1)[-1]
                if ("isi", artifact_id) in available:
                    self.isi_saved_box.addItem(
                        f"{record.execution_id[:8]} · {record.operation_id}", record.execution_id
                    )

    def open_saved_waveform_result(self) -> None:
        execution_id = self.waveform_saved_box.currentData()
        record = next((item for item in self._execution_records if item.execution_id == execution_id), None)
        if record is not None:
            self._current_waveform_record = record
            self._show_execution_provenance(record)
            self._load_waveform_record(record)

    def open_saved_isi_result(self) -> None:
        execution_id = self.isi_saved_box.currentData()
        record = next((item for item in self._execution_records if item.execution_id == execution_id), None)
        if record is not None:
            self._current_isi_record = record
            self._show_execution_provenance(record)
            self._load_isi_record(record)

    def _show_execution_provenance(self, record: ExecutionRecord) -> None:
        """Show compact historical provenance without implying a rerun."""
        lines = [
            "Historical scientific execution — restored from project; not rerun.",
            f"Operation: {record.operation_id} v{record.operation_version}",
            f"Implementation: {record.implementation_version}",
            f"Execution ID: {record.execution_id}",
            f"Status: {record.status.value}; finished {record.finished_at_utc.isoformat()}",
            f"Approval: {record.provenance.approved_by} · {record.provenance.approval_id} · {record.provenance.approved_at_utc.isoformat()}",
            f"Dataset identity: {record.input_reference.dataset_id if record.input_reference else 'unknown'}",
            f"Channels: {record.selected_channels}",
            f"Sample interval: {record.sample_interval.model_dump(mode='json') if record.sample_interval else 'not specified'}",
        ]
        if record.parameters is not None:
            lines.append(f"Parameters: {json.dumps(record.parameters.model_dump(mode='json'), ensure_ascii=False)}")
        for item in record.input_metadata:
            lines.append(f"Input metadata {item.name}: {item.value!r} [{item.status.value}]")
        if record.output is not None:
            output = record.output.model_dump(mode="json")
            for key in (
                "artifact_reference", "interval_artifact_reference", "accepted_event_count",
                "excluded_event_count", "waveform_shape", "event_count", "interval_count",
                "polarity_mode", "sampling_rate_hz", "sampling_rate_status",
            ):
                if key in output:
                    lines.append(f"Output {key}: {output[key]}")
        lines.extend(f"Warning: {warning}" for warning in record.warnings)
        self.operation_review.setPlainText("\n".join(lines))

    def prepare_spike_detection(self) -> None:
        """Validate and prepare the exact visible-channel detector request."""
        dataset = self.active_dataset
        registration = self.active_registration
        navigation = self.navigation_state
        if dataset is None or registration is None or navigation is None:
            return
        channel = self.channel_box.currentData()
        if channel is None:
            QMessageBox.warning(self, "Analysis not prepared", "Select one analysis channel.")
            return
        interval = navigation.visible_window
        sample_interval = SampleInterval(start=interval.start, stop=interval.stop)
        reference = registration.reference((int(channel),), sample_interval)
        detector_id = self.detector_box.currentData()
        try:
            if detector_id == NEGATIVE_DETECTOR_ID:
                operation_input = NegativeSpikeInput(signal=reference)
                parameters = NegativeSpikeParameters(
                    threshold_sd=self._float_parameter(
                        "threshold_sd", self.threshold.text()
                    )
                )
            else:
                mode = self.signed_mode_box.currentData()
                value = self._float_parameter(
                    "signed threshold or multiplier", self.signed_threshold.text()
                )
                if mode == "absolute":
                    noise_text = self.noise_override.text().strip()
                    operation_parameters = {
                        "threshold_mode": "absolute",
                        "threshold": value,
                        "noise_override": (
                            self._float_parameter("noise override", noise_text)
                            if noise_text
                            else None
                        ),
                    }
                else:
                    operation_parameters = {
                        "threshold_mode": "noise_multiplier",
                        "stdmin": value,
                    }
                operation_input = SignedPeakInput(signal=reference)
                parameters = SignedPeakParameters(**operation_parameters)

            summary_metadata = dataset.exploration.summary.metadata
            metadata_names = (
                "signal_kind",
                "sampling_rate_hz",
                "units",
                "filtering_description",
            )
            snapshots = tuple(
                MetadataSnapshot(
                    name=name,
                    value=summary_metadata[name].value,
                    status=summary_metadata[name].status,
                )
                for name in metadata_names
            )
            context = OperationContext(
                dataset_reference=registration.dataset,
                metadata=snapshots,
                selected_channels=(int(channel),),
                sample_interval=sample_interval,
            )
            definition = self.operation_registry.get(detector_id)
            prepared = self.operation_runner.prepare(
                detector_id,
                definition.version,
                operation_input,
                parameters,
                context=context,
            )
        except (TypeError, ValueError, KeyError) as exc:
            self._prepared_execution = None
            self.operation_review.setPlainText(f"Preparation failed:\n{exc}")
            self.analysis_state_label.setText("Failed — correct the inputs and prepare again.")
            self._update_action_state()
            return

        self._prepared_execution = prepared
        metadata_lines = [
            f"{item.name}: {item.value!r} [{item.status.value}]"
            for item in snapshots
        ]
        parameter_snapshot = parameters.model_dump(mode="json")
        warning_lines = list(dict.fromkeys(definition.warnings + prepared.warnings))
        review = [
            f"Operation: {definition.operation_id} v{definition.version}",
            f"Implementation: {definition.implementation_version}",
            f"Dataset: {registration.dataset.dataset_id}",
            f"Selected channel (0-based): {channel}",
            f"Original sample interval: [{interval.start}, {interval.stop})",
            f"Parameters: {parameter_snapshot}",
            "Metadata snapshot:",
            *metadata_lines,
            "Expected output: typed spike events, sample indices, amplitudes, noise estimate and threshold.",
            "Scientific warnings and limitations:",
            *(warning_lines or ["None from preparation."]),
            *[f"Limitation: {item}" for item in definition.limitations],
        ]
        if self.store is None:
            review.extend(
                ["", "Execution requires an explicitly selected or created project."]
            )
        self.operation_review.setPlainText("\n".join(review))
        self.analysis_state_label.setText(
            "Awaiting approval — review the prepared operation before approving."
        )
        self._update_action_state()

    @staticmethod
    def _float_parameter(name: str, text: str) -> float:
        try:
            return float(text.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric value.") from exc

    @staticmethod
    def _integer_parameter(name: str, text: str) -> int:
        try:
            value = int(text.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer sample count.") from exc
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return value

    def _metadata_context_for_reference(
        self, reference: DataReference
    ) -> OperationContext:
        if self.active_dataset is None or self.active_registration is None:
            raise ValueError("a dataset must be active before preparing an operation.")
        metadata = self.active_dataset.exploration.summary.metadata
        fields = tuple(
            MetadataSnapshot(name=name, value=metadata[name].value, status=metadata[name].status)
            for name in ("sampling_rate_hz", "units")
        )
        return OperationContext(
            dataset_reference=self.active_registration.dataset,
            metadata=fields,
            selected_channels=reference.channels,
            sample_interval=reference.sample_interval,
        )

    def _set_prepared_review(self, prepared: PreparedExecution, lines: list[str]) -> None:
        self._prepared_execution = prepared
        self._prepared_mua_batch = None
        self.operation_review.setPlainText("\n".join(lines))
        self.analysis_state_label.setText(
            "Awaiting approval — review the prepared operation before approving."
        )
        self._update_action_state()

    def prepare_waveform_extraction(self) -> None:
        if self.store is None or self.active_registration is None:
            QMessageBox.information(self, "Project required", "Select a project before preparing waveform extraction.")
            return
        detector_id = self.waveform_source_box.currentData()
        detector_record = self._detector_records.get(detector_id)
        if detector_record is None or detector_record.status is not ExecutionStatus.SUCCEEDED:
            QMessageBox.warning(self, "Detector result unavailable", "Select a successful detector execution.")
            return
        try:
            detections = detection_collection_from_record(detector_record)
            channel_filter = self.waveform_channel_filter.currentData()
            selected_event_id = None
            if self.waveform_only_selected_event.isChecked():
                row = self.event_table.currentRow()
                if row < 0 or row >= len(self._current_events) or self._current_record is not detector_record:
                    raise ValueError("select an event from the chosen detector execution first.")
                selected_event_id = str(self._current_events[row]["event_id"])
            selected_events = tuple(
                event
                for event in detections.events
                if (channel_filter is None or event.channel_index == channel_filter)
                and (selected_event_id is None or event.event_id == selected_event_id)
            )
            if not selected_events:
                raise ValueError("the selected channel/event filters contain no detected events.")
            detections = detections.model_copy(update={"events": selected_events})
            pre_samples = self._integer_parameter("pre-event samples", self.waveform_pre_samples.text())
            post_samples = self._integer_parameter("post-event samples", self.waveform_post_samples.text())
            source_reference = detector_record.data_references[0]
            channels = source_reference.channels
            full_reference = self.active_registration.reference(
                channels,
                SampleInterval(start=0, stop=self.active_registration.sample_count),
            )
            operation_input = WaveformExtractionInput(
                signal=full_reference,
                detections=detections,
            )
            context = self._metadata_context_for_reference(full_reference)
            definition = self.operation_registry.get("neuroscope.spike.extract-waveforms")
            parameters = WaveformExtractionParameters(
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            prepared = self.operation_runner.prepare(
                definition.operation_id,
                definition.version,
                operation_input,
                parameters,
                context=context,
            )
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            self._prepared_execution = None
            self.operation_review.setPlainText(f"Waveform preparation failed:\n{exc}")
            self.analysis_state_label.setText("Failed — correct waveform inputs and prepare again.")
            self._update_action_state()
            return
        self._set_prepared_review(
            prepared,
            [
                f"Operation: {definition.operation_id} v{definition.version}",
                f"Implementation: {definition.implementation_version}",
                f"Dataset: {full_reference.dataset.dataset_id}",
                f"Detector execution: {detector_record.execution_id} ({detector_record.operation_id} v{detector_record.operation_version})",
                f"Detection reference: {detections.detection_reference}",
                f"Channels: {detections.source_data_reference.channels}; events selected: {len(selected_events)}",
                f"Pre/post samples: {pre_samples}/{post_samples}; waveform length: {parameters.waveform_length}",
                "Boundary policy: exclude complete snippets outside the recording; no truncation or padding.",
                f"Sampling rate: {detections.sampling_rate_hz!r} [{detections.sampling_rate_status.value}]",
                f"Amplitude units: {detections.amplitude_units!r} [{detections.amplitude_units_status.value}]",
                "Alignment: original detector sample; no realignment or preprocessing.",
                *[f"Warning: {item}" for item in definition.warnings],
                *[f"Limitation: {item}" for item in definition.limitations],
            ],
        )

    def _on_waveform_parameter_changed(self, _value: str) -> None:
        pre_text = self.waveform_pre_samples.text().strip()
        post_text = self.waveform_post_samples.text().strip()
        try:
            length = self._integer_parameter("pre-event samples", pre_text) + self._integer_parameter(
                "post-event samples", post_text
            ) + 1
            self.waveform_length_label.setText(f"Waveform length: {length} samples")
        except ValueError:
            self.waveform_length_label.setText("Waveform length: — (enter non-negative integer windows)")
        self._invalidate_prepared("waveform parameters changed")
        if self._current_waveform_record is not None:
            self.waveform_display_label.setText("Stale waveform result — prior extraction retained.")

    def _on_waveform_selection_changed(self, _value) -> None:
        self._invalidate_prepared("waveform source or event selection changed")
        if self._current_waveform_record is not None:
            self.waveform_display_label.setText("Stale waveform result — selection changed.")

    def _on_waveform_source_changed(self, _index: int) -> None:
        current_id = self.waveform_source_box.currentData()
        self.waveform_channel_filter.blockSignals(True)
        self.waveform_channel_filter.clear()
        self.waveform_channel_filter.addItem("All detected channels", None)
        record = self._detector_records.get(current_id)
        if record is not None:
            try:
                events = detection_collection_from_record(record).events
                for channel in sorted({event.channel_index for event in events}):
                    self.waveform_channel_filter.addItem(f"Channel {channel}", channel)
            except ValueError:
                pass
        self.waveform_channel_filter.blockSignals(False)
        self._on_waveform_selection_changed(None)

    def _on_isi_source_changed(self, _index: int) -> None:
        self._invalidate_prepared("ISI detector execution changed")
        if self._current_isi_record is not None:
            self.isi_display_label.setText("Stale ISI result — detector selection changed.")

    def _on_isi_parameters_changed(self, _value) -> None:
        self._invalidate_prepared("ISI parameters changed")
        if self._current_isi_record is not None:
            self.isi_display_label.setText("Stale ISI result — parameters changed.")

    def _on_isi_threshold_toggled(self, state: int) -> None:
        enabled = bool(state)
        self.isi_threshold.setEnabled(enabled)
        self.isi_threshold_units.setEnabled(enabled)
        self._on_isi_parameters_changed(state)

    def prepare_isi_analysis(self) -> None:
        if self.store is None:
            QMessageBox.information(self, "Project required", "Select a project before preparing ISI analysis.")
            return
        detector_id = self.isi_source_box.currentData()
        detector_record = self._detector_records.get(detector_id)
        if detector_record is None or detector_record.status is not ExecutionStatus.SUCCEEDED:
            QMessageBox.warning(self, "Detector result unavailable", "Select a successful detector execution.")
            return
        try:
            if (
                not detector_record.data_references
                or self.active_registration is None
                or detector_record.data_references[0].session_handle
                != self.active_registration.session_handle
            ):
                raise ValueError("the selected detector execution belongs to an inactive dataset registration.")
            collection = isi_collection_from_record(detector_record)
            mode = self.isi_polarity_box.currentData()
            if mode not in {"separate", "combine"}:
                raise ValueError("select whether polarities are analyzed separately or combined.")
            threshold_samples = threshold_ms = None
            if self.isi_threshold_enabled.isChecked():
                value = self._float_parameter("short-interval threshold", self.isi_threshold.text())
                unit = self.isi_threshold_units.currentData()
                if unit == "samples":
                    if not value.is_integer() or value < 0:
                        raise ValueError("sample threshold must be a non-negative integer.")
                    threshold_samples = int(value)
                else:
                    threshold_ms = value
            parameters = ISIParameters(
                polarity_mode=mode,
                short_interval_threshold_samples=threshold_samples,
                short_interval_threshold_ms=threshold_ms,
            )
            operation_input = ISIInput(events=collection)
            context = OperationContext(
                dataset_reference=collection.dataset,
                metadata=(MetadataSnapshot(
                    name="sampling_rate_hz",
                    value=collection.sampling_rate_hz,
                    status=collection.sampling_rate_status,
                ),),
                selected_channels=collection.source_channels,
                sample_interval=collection.source_sample_interval,
            )
            definition = self.operation_registry.get("neuroscope.spike.analyze-isi")
            prepared = self.operation_runner.prepare(
                definition.operation_id,
                definition.version,
                operation_input,
                parameters,
                context=context,
            )
        except (TypeError, ValueError, KeyError) as exc:
            self._prepared_execution = None
            self.operation_review.setPlainText(f"ISI preparation failed:\n{exc}")
            self.analysis_state_label.setText("Failed — correct ISI inputs and prepare again.")
            self._update_action_state()
            return
        self._set_prepared_review(
            prepared,
            [
                f"Operation: {definition.operation_id} v{definition.version}",
                f"Implementation: {definition.implementation_version}",
                f"Dataset: {collection.dataset.dataset_id}",
                f"Detector execution: {detector_record.execution_id} ({detector_record.operation_id} v{detector_record.operation_version})",
                f"Events: {len(collection.events)}; channels: {collection.source_channels}",
                f"Polarity grouping: {mode}",
                f"Short-interval threshold: {threshold_samples!r} samples; {threshold_ms!r} ms",
                f"Sampling rate: {collection.sampling_rate_hz!r} [{collection.sampling_rate_status.value}]",
                "Statistics use population standard deviation (ddof=0). Short intervals are descriptive, not proven refractory violations.",
                *[f"Warning: {item}" for item in prepared.warnings],
                *[f"Warning: {item}" for item in definition.warnings],
                *[f"Limitation: {item}" for item in definition.limitations],
            ],
        )

    def _on_mua_parameters_changed(self, _value=None) -> None:
        self._invalidate_prepared("calcMUA selection or parameters changed")
        if self._mua_records:
            self.mua_display_label.setText(
                "Stale calcMUA result retained — selection or parameters changed."
            )

    def _on_mua_average_selection_changed(self, _value=None) -> None:
        self._invalidate_prepared("MUA averaging inputs changed")
        self._update_action_state()

    def _register_mua_result(self, record: ExecutionRecord, *, average: bool) -> None:
        self._mua_records[record.execution_id] = record
        output = record.output
        if isinstance(output, (CalcMUAOutput, MUAAverageOutput)):
            self._mua_outputs[record.execution_id] = output
            self._mua_provenance_snapshots[record.execution_id] = record.to_snapshot()
        if not average:
            self._mua_input_references[record.execution_id] = (
                mua_artifact_reference_from_record(record)
            )
        label = ""
        if isinstance(output, CalcMUAOutput):
            label = f"Channel {output.source_channel} · calcMUA · {output.mode}"
        elif isinstance(output, MUAAverageOutput):
            label = f"Average · {output.contributing_channels} channels"
        if not label:
            return
        self._add_mua_result_entry(
            record.execution_id, label, average=average, displayed=True
        )

    def _add_mua_result_entry(
        self, execution_id: str, label: str, *, average: bool, displayed: bool
    ) -> None:
        widgets = (self.mua_display_results,) if average else (
            self.mua_display_results,
            self.mua_average_inputs,
        )
        for widget in widgets:
            if any(
                str(widget.item(index).data(Qt.UserRole)) == execution_id
                for index in range(widget.count())
            ):
                continue
            widget.blockSignals(True)
            item = QListWidgetItem(f"{label} · {execution_id[:8]}")
            item.setData(Qt.UserRole, execution_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if widget is self.mua_display_results:
                item.setCheckState(Qt.Checked if displayed else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Unchecked)
            widget.addItem(item)
            widget.blockSignals(False)

    def _refresh_mua_project_results(self) -> None:
        """Expose valid persisted MUA artifacts after opening a project."""
        self._mua_records.clear()
        self._mua_outputs.clear()
        self._mua_input_references.clear()
        self._mua_provenance_snapshots.clear()
        self._mua_selection_views = []
        self._mua_artifact_view = None
        self._current_mua_record = None
        self.mua_display_results.blockSignals(True)
        self.mua_display_results.clear()
        self.mua_display_results.blockSignals(False)
        self.mua_average_inputs.blockSignals(True)
        self.mua_average_inputs.clear()
        self.mua_average_inputs.blockSignals(False)
        if self.store is None:
            return
        verified_mua_references = {
            f"artifact://mua/{item.get('artifact_id')}"
            for item in self.store.discover_artifacts()
            if item.get("kind") == "mua" and item.get("status") == "available"
        }
        for snapshot in self.store.load_execution_history():
            if snapshot.get("status") != "succeeded":
                continue
            execution_id = str(snapshot.get("execution_id", ""))
            try:
                if snapshot.get("operation_id") == CALC_MUA_ID:
                    output = CalcMUAOutput.model_validate_json(
                        json.dumps(snapshot.get("output"), allow_nan=False)
                    )
                    if output.artifact_reference not in verified_mua_references:
                        raise ValueError("artifact lineage is missing, invalid, or unsupported.")
                    reference = mua_artifact_reference_from_snapshot(snapshot)
                    with self.store.mua_artifacts.open(output.artifact_reference):
                        pass
                    self._mua_input_references[execution_id] = reference
                    self._mua_outputs[execution_id] = output
                    self._mua_provenance_snapshots[execution_id] = snapshot
                    self._add_mua_result_entry(
                        execution_id,
                        f"Channel {output.source_channel} · calcMUA · {output.mode}",
                        average=False,
                        displayed=False,
                    )
                elif snapshot.get("operation_id") == AVERAGE_MUA_OPERATION.operation_id:
                    output = MUAAverageOutput.model_validate_json(
                        json.dumps(snapshot.get("output"), allow_nan=False)
                    )
                    if output.artifact_reference not in verified_mua_references:
                        raise ValueError("artifact lineage is missing, invalid, or unsupported.")
                    with self.store.mua_artifacts.open(output.artifact_reference):
                        pass
                    self._mua_outputs[execution_id] = output
                    self._mua_provenance_snapshots[execution_id] = snapshot
                    self._add_mua_result_entry(
                        execution_id,
                        f"Average · {output.contributing_channels} channels",
                        average=True,
                        displayed=False,
                    )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                self.operation_review.append(
                    f"\nHistorical MUA result {execution_id} unavailable: {exc}"
                )
                continue

    def _on_mua_display_changed(self, _value=None) -> None:
        """Visibility changes affect only rendering and never invalidate approval."""
        entries = self._selected_mua_display_entries()
        if not entries or self.store is None:
            self._mua_selection_views = []
            self._render_mua_view()
            return
        if len(entries) == 1:
            record = self._mua_records.get(str(entries[0]["execution_id"]))
            if record is not None and isinstance(record.output, CalcMUAOutput):
                self._current_mua_record = record
                self._load_mua_record(record)
                return
        self._mua_artifact_view = None
        self.mua_display_label.setText("Reading selected MUAtrace artifacts in a background worker…")
        task = lambda: _read_mua_selection(
            self.store.mua_artifacts.root, tuple(entries), 1400
        )
        self._start_view_worker("mua", task)

    def _on_mua_layout_changed(self, _value=None) -> None:
        """Change only the presentation of already loaded artifact views."""
        self._render_mua_view()

    def _selected_mua_display_entries(self) -> list[dict[str, object]]:
        entries = []
        for index in range(self.mua_display_results.count()):
            item = self.mua_display_results.item(index)
            if item.checkState() != Qt.Checked:
                continue
            execution_id = str(item.data(Qt.UserRole))
            output = self._mua_outputs.get(execution_id)
            if output is None:
                continue
            if isinstance(output, CalcMUAOutput):
                entries.append(
                    {
                        "execution_id": execution_id,
                        "label": f"Channel {output.source_channel} · calcMUA · {output.mode}",
                        "artifact_reference": output.artifact_reference,
                        "sample_start": output.sample_interval.start,
                        "sampling_rate_hz": output.sampling_rate_hz,
                        "sampling_rate_status": output.sampling_rate_status,
                        "units": output.mua_units,
                        "mode": output.mode,
                        "is_average": False,
                    }
                )
            elif isinstance(output, MUAAverageOutput):
                entries.append(
                    {
                        "execution_id": execution_id,
                        "label": f"Average · {output.contributing_channels} channels",
                        "artifact_reference": output.artifact_reference,
                        "sample_start": output.sample_interval.start,
                        "sampling_rate_hz": output.sampling_rate_hz,
                        "sampling_rate_status": output.sampling_rate_status,
                        "units": output.mua_units,
                        "mode": output.mode,
                        "is_average": True,
                    }
                )
        return entries

    def prepare_mua_average(self) -> None:
        if self.store is None:
            QMessageBox.information(self, "Project required", "Select a project before preparing an average.")
            return
        try:
            selected_refs = []
            for index in range(self.mua_average_inputs.count()):
                item = self.mua_average_inputs.item(index)
                if item.checkState() == Qt.Checked:
                    reference = self._mua_input_references.get(str(item.data(Qt.UserRole)))
                    if reference is None:
                        raise ValueError("a selected MUAtrace execution is unavailable.")
                    selected_refs.append(reference)
            input_model = MUAAverageInput(
                traces=tuple(selected_refs)
            )
            definition = self.operation_registry.get(AVERAGE_MUA_OPERATION.operation_id)
            parameters = MUAAverageParameters()
            prepared = self.operation_runner.prepare(
                definition.operation_id,
                definition.version,
                input_model,
                parameters,
            )
        except (TypeError, ValueError, KeyError) as exc:
            self._prepared_execution = None
            self._prepared_mua_batch = None
            self.operation_review.setPlainText(f"MUA average preparation failed:\n{exc}")
            self.analysis_state_label.setText("Failed — select compatible traces and prepare again.")
            self._update_action_state()
            return
        self._set_prepared_review(
            prepared,
            [
                f"Operation: {definition.operation_id} v{definition.version}",
                f"Implementation: {definition.implementation_version}",
                f"Method: {prepared.parameters.method}",
                f"Selected artifact count: {len(input_model.traces)}",  # type: ignore[attr-defined]
                *[
                    f"Channel {trace.channel_index}; execution {trace.execution_id}; artifact {trace.artifact_reference}; dataset {trace.dataset_id}"
                    for trace in input_model.traces
                ],
                "Compatibility: identical dataset, sample interval/grid, confirmed sampling rate, calcMUA implementation and parameters, units, signal metadata, and filtering metadata.",
                "Nonfinite input traces are rejected; channels and samples will not be dropped or interpolated.",
                "The result is a new equal-channel arithmetic-mean artifact; source artifacts are unchanged.",
                *[f"Warning: {warning}" for warning in definition.warnings],
                *[f"Limitation: {limitation}" for limitation in definition.limitations],
            ],
        )

    def _inspect_mua_provenance(self) -> None:
        ids = [
            str(self.mua_display_results.item(i).data(Qt.UserRole))
            for i in range(self.mua_display_results.count())
            if self.mua_display_results.item(i).checkState() == Qt.Checked
        ]
        if not ids and self._current_mua_record is not None:
            ids = [self._current_mua_record.execution_id]
        snapshots = [self._mua_provenance_snapshots[item] for item in ids if item in self._mua_provenance_snapshots]
        QMessageBox.information(
            self,
            "MUAtrace provenance",
            __import__("json").dumps(snapshots, indent=2, ensure_ascii=False),
        )

    def prepare_calc_mua(self) -> None:
        if self.store is None or self.active_registration is None or self.navigation_state is None:
            QMessageBox.information(
                self,
                "Dataset and project required",
                "Load a dataset and select a project before preparing calcMUA.",
            )
            return
        channels = [
            int(self.mua_calculation_channels.item(i).data(Qt.UserRole))
            for i in range(self.mua_calculation_channels.count())
            if self.mua_calculation_channels.item(i).checkState() == Qt.Checked
        ]
        # Backward-compatible single-channel entry point for callers using the prior combo.
        if not channels and self.mua_channel_box.currentData() is not None:
            channels = [int(self.mua_channel_box.currentData())]
        if not channels:
            self.operation_review.setPlainText(
                "calcMUA preparation failed:\nSelect one or more channels to calculate."
            )
            return
        mode = self.mua_mode_box.currentData()
        if mode not in {"rel", "abs"}:
            self.operation_review.setPlainText("calcMUA preparation failed:\nSelect rel or abs output mode.")
            return
        try:
            stdmin = self._float_parameter("stdmin", self.mua_stdmin.text())
            bandwidth = self._float_parameter("muawin bandwidth in seconds", self.mua_bandwidth.text())
            num_points = self._integer_parameter("num_points", self.mua_num_points.text())
            if num_points <= 0:
                raise ValueError("num_points must be a positive integer.")
            visible = self.navigation_state.visible_window
            sample_interval = SampleInterval(start=visible.start, stop=visible.stop)
            batch_id = str(uuid4())
            metadata = self.active_dataset.exploration.summary.metadata
            metadata_fields = tuple(
                MetadataSnapshot(name=name, value=metadata[name].value, status=metadata[name].status)
                for name in (
                    "signal_kind",
                    "sampling_rate_hz",
                    "units",
                    "filtering_description",
                )
            )
            parameters = CalcMUAParameters(
                stdmin=stdmin,
                mua_window_s=bandwidth,
                mode=mode,
                num_points=num_points,
            )
            definition = self.operation_registry.get(CALC_MUA_ID)
            prepared_batch = []
            references = []
            for channel in channels:
                reference = self.active_registration.reference((channel,), sample_interval)
                context = OperationContext(
                    dataset_reference=reference.dataset,
                    metadata=metadata_fields,
                    selected_channels=(channel,),
                    sample_interval=sample_interval,
                    reproducibility=ReproducibilityInfo(
                        configuration_reference=f"mua-calc-batch://{batch_id}"
                    ),
                )
                prepared_batch.append(
                    self.operation_runner.prepare(
                        CALC_MUA_ID,
                        definition.version,
                        CalcMUAInput(signal=reference),
                        parameters,
                        context=context,
                    )
                )
                references.append(reference)
        except (TypeError, ValueError, KeyError) as exc:
            self._prepared_execution = None
            self._prepared_mua_batch = None
            self.operation_review.setPlainText(f"calcMUA preparation failed:\n{exc}")
            self.analysis_state_label.setText("Failed — correct calcMUA inputs and prepare again.")
            self._update_action_state()
            return
        if len(prepared_batch) == 1:
            prepared = prepared_batch[0]
            reference = references[0]
            channel = channels[0]
        else:
            self._prepared_execution = None
            self._prepared_mua_batch = tuple(prepared_batch)
            lines = [
                f"Batch: mua-calc-batch://{batch_id}",
                f"Operation: {definition.operation_id} v{definition.version}",
                f"Implementation: {definition.implementation_version}",
                f"Independent channel operations: {len(prepared_batch)}",
                f"Dataset: {references[0].dataset.dataset_id}",
                f"Original sample interval: [{sample_interval.start}, {sample_interval.stop})",
                f"Shared parameters: stdmin={stdmin}; muawin={bandwidth} s; mode={mode}; num_points={num_points}",
                "Each selected channel is separately validated, approved, executed, and recorded. Partial failures remain visible per channel.",
            ]
            for channel, reference, item in zip(channels, references, prepared_batch, strict=True):
                lines.extend(
                    [
                        f"Channel {channel}: {reference.dataset.dataset_id}, samples [{sample_interval.start}, {sample_interval.stop})",
                        *[f"  Metadata {meta.name}: {meta.value!r} [{meta.status.value}]" for meta in metadata_fields],
                        *[f"  Warning: {warning}" for warning in item.warnings],
                    ]
                )
            self.operation_review.setPlainText("\n".join(lines))
            self.analysis_state_label.setText(
                "Awaiting approval — review every independently prepared channel operation."
            )
            self._update_action_state()
            return
        self._set_prepared_review(
            prepared,
            [
                f"Operation: {definition.operation_id} v{definition.version}",
                f"Implementation: {definition.implementation_version}",
                "Scientific method: developed by Dr. Cecilia Herbert",
                "MATLAB implementation: independently written by Fiamma Liz Leites as calcmua_verFi.m",
                "Python implementation: NeuroScope's separate implementation based on Leites' MATLAB implementation; parity not established",
                f"Recorded MATLAB source SHA-256 (not independently reverified here): {CALC_MUA_SOURCE_SHA256}",
                f"Dataset: {reference.dataset.dataset_id}; channel (0-based): {channel}",
                f"Selected original samples: [{sample_interval.start}, {sample_interval.stop})",
                *[
                    f"Metadata {item.name}: {item.value!r} [{item.status.value}]"
                    for item in metadata_fields
                ],
                f"stdmin: {stdmin}; muawin: {bandwidth} s; mode: {mode}; num_points: {num_points}",
                "Noise: median(abs(signal))/0.6745; signed threshold; prominence 0.75×|threshold|; strict peak cap 25×noise.",
                "MUA output: normal-kernel KDE; rel is PDF (s^-1), abs is event-count-scaled density (events s^-1).",
                "No filtering, resampling, or normalization will be applied. Times are relative to the selected window; no acquisition origin is asserted.",
                *[f"Warning: {item}" for item in prepared.warnings],
                *[f"Warning: {item}" for item in definition.warnings],
                *[f"Limitation: {item}" for item in definition.limitations],
            ],
        )

    def _start_view_worker(self, view_name: str, task) -> None:
        self._view_generations[view_name] += 1
        generation = self._view_generations[view_name]
        thread = QThread(self)
        worker = _ViewWorker(view_name, generation, task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_view_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda t=thread, w=worker: self._cleanup_view_worker(t, w))
        self._view_workers.append((thread, worker))
        thread.start()

    def _cleanup_view_worker(self, thread: QThread, worker: _ViewWorker) -> None:
        self._view_workers = [pair for pair in self._view_workers if pair[0] is not thread]
        thread.deleteLater()

    def _on_view_finished(self, name: str, generation: int, result, error) -> None:
        if generation != self._view_generations.get(name):
            return
        if error is not None:
            label = (
                self.waveform_display_label
                if name == "waveform"
                else self.isi_display_label
                if name == "isi"
                else self.mua_display_label
            )
            label.setText(f"Could not load {name} artifact: {error}")
            return
        if name == "waveform":
            self._waveform_artifact_view = result
            self._populate_waveform_event_picker(result)
            self._populate_waveform_excluded_table(result)
            self._render_waveform_view()
            record = self._current_waveform_record
            source_id = record.input_reference.dataset_id if record and record.input_reference else None
            if self.active_registration is None or source_id != self.active_registration.dataset.dataset_id:
                self.waveform_display_label.setText(
                    "Historical saved waveform artifact; source navigation is disabled."
                )
        elif name == "isi":
            self._isi_artifact_view = result
            self._render_isi_view()
        else:
            if isinstance(result, list):
                self._mua_selection_views = result
                self._mua_artifact_view = None
            else:
                self._mua_artifact_view = result
                self._mua_selection_views = []
            self._render_mua_view()

    def _refresh_detector_execution_choices(self) -> None:
        for combo in (self.waveform_source_box, self.isi_source_box):
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select detector execution", None)
            records = sorted(
                self._detector_records.values(),
                key=lambda item: item.finished_at_utc,
            )
            for record in records:
                reference = record.data_references[0]
                stale = (
                    self.active_registration is None
                    or reference.dataset.dataset_id != self.active_registration.dataset.dataset_id
                )
                label = (
                    f"{record.operation_id} — {record.execution_id[:8]}"
                    + (" (historical dataset)" if stale else "")
                )
                combo.addItem(label, record.execution_id)
            if previous is not None:
                index = combo.findData(previous)
                if index >= 0:
                    combo.setCurrentIndex(index)
                elif self._current_record is not None:
                    combo.setCurrentIndex(combo.findData(self._current_record.execution_id))
            elif self._current_record is not None:
                combo.setCurrentIndex(combo.findData(self._current_record.execution_id))
            combo.blockSignals(False)
        self._on_waveform_source_changed(self.waveform_source_box.currentIndex())
        self._update_action_state()

    def _load_isi_record(self, record: ExecutionRecord) -> None:
        if self.store is None or record.output is None:
            return
        output = record.output
        if not isinstance(output, ISIOutput):
            self.isi_display_label.setText("ISI execution has an incompatible output type.")
            return
        self._isi_artifact_view = None
        source_id = record.input_reference.dataset_id if record.input_reference else None
        source_active = (
            source_id is not None and self.active_registration is not None
            and self.active_registration.dataset.dataset_id == source_id
        )
        if not source_active:
            self.isi_display_label.setText(
                "Historical ISI result — displaying its saved interval artifact; source navigation is disabled."
            )
        self.isi_group_box.blockSignals(True)
        self.isi_group_box.clear()
        for summary in output.summaries:
            key = (summary.channel_index, summary.polarity_group)
            self.isi_group_box.addItem(
                f"Channel {summary.channel_index} · {summary.polarity_group}", key
            )
        if self.isi_group_box.count() and self.isi_group_box.currentIndex() < 0:
            self.isi_group_box.setCurrentIndex(0)
        self.isi_group_box.blockSignals(False)
        self.workspace_tabs.setCurrentIndex(2)
        self._reload_isi_view()

    def _load_mua_record(self, record: ExecutionRecord) -> None:
        if self.store is None or record.output is None:
            return
        output = record.output
        if not isinstance(output, CalcMUAOutput):
            self.mua_display_label.setText("calcMUA execution has an incompatible output type.")
            return
        reference = record.data_references[0] if record.data_references else None
        active = (
            reference is not None
            and self.active_registration is not None
            and reference.dataset.dataset_id == self.active_registration.dataset.dataset_id
        )
        exploration = self.active_dataset.exploration if active and self.active_dataset is not None else None
        self._mua_artifact_view = None
        self.workspace_tabs.setCurrentIndex(3)
        if not active:
            self.mua_display_label.setText(
                "Historical/stale MUA result. Its source recording is not active; showing the stored trace only."
            )
        else:
            self.mua_display_label.setText("Reading calcMUA trace and bounded source overlay…")
        task = lambda: _read_mua_view(
            self.store.mua_artifacts.root,
            output.artifact_reference,
            exploration,
            output.source_channel,
            1400,
        )
        self._start_view_worker("mua", task)

    def _render_mua_view(self) -> None:
        self.mua_figure.clear()
        if self._mua_selection_views:
            separate = self.mua_layout_box.currentData() == "separate"
            items = self._mua_selection_views
            shared_grid = all(
                item["interval"] == items[0]["interval"]
                and item["sampling_rate_hz"] == items[0]["sampling_rate_hz"]
                and item["sampling_rate_status"] == items[0]["sampling_rate_status"]
                for item in items[1:]
            )
            separate = separate or not shared_grid
            if separate:
                axes = [
                    self.mua_figure.add_subplot(
                        len(items), 1, index + 1,
                        sharex=(self.mua_figure.axes[0] if index and shared_grid else None),
                    )
                    for index in range(len(items))
                ]
            else:
                axes = [self.mua_figure.add_subplot(111)]
            for index, item in enumerate(items):
                axis = axes[index] if separate else axes[0]
                indices = np.asarray(item["indices"])
                rate = (
                    float(item["sampling_rate_hz"])
                    if item["sampling_rate_status"] is MetadataStatus.CONFIRMED
                    else None
                )
                x_values = indices / rate if rate else indices
                axis.plot(x_values, item["values"], linewidth=1.1, label=str(item["label"]))
                axis.set_ylabel(f"MUA ({item['units']})")
                if separate:
                    axis.set_title(str(item["label"]))
                axis.legend(loc="best")
            rate = items[0]["sampling_rate_hz"]
            confirmed_rate = (
                float(rate) if items[0]["sampling_rate_status"] is MetadataStatus.CONFIRMED else None
            )
            axes[-1].set_xlabel(
                "Time relative to original recording sample zero (s; no acquisition origin asserted)"
                if confirmed_rate else "Original recording sample index (0-based)"
            )
            if not shared_grid:
                self.mua_display_label.setText(
                    "Selected traces have different sample/time grids; separate panels use their own coordinates."
                )
            else:
                self.mua_display_label.setText(
                    f"Displaying {len(items)} selected MUAtrace artifact(s); "
                    f"layout: {'separate panels' if separate else 'overlay'}. Visibility changes do not recalculate results."
                )
            self.mua_figure.tight_layout()
            self.mua_canvas.draw_idle()
            return
        axis = self.mua_figure.add_subplot(111)
        view = self._mua_artifact_view
        record = self._current_mua_record
        if view is None or record is None or not isinstance(record.output, CalcMUAOutput):
            axis.text(0.5, 0.5, "No calcMUA result selected.", ha="center", va="center", transform=axis.transAxes)
            self.mua_canvas.draw_idle()
            return
        output = record.output
        rate = output.sampling_rate_hz if output.sampling_rate_status is MetadataStatus.CONFIRMED else None
        source_indices = np.asarray(view["source_sample_indices"])
        source_values = np.asarray(view["source_values"])
        trace_indices = np.asarray(view["trace_sample_indices"])
        trace_values = np.asarray(view["trace_values"])
        source_x = source_indices / rate if rate is not None else source_indices
        trace_x = trace_indices / rate if rate is not None else trace_indices
        if source_indices.size:
            axis.plot(source_x, source_values, color="0.35", linewidth=0.7, label="Source signal")
        units = output.amplitude_units
        units_status = output.amplitude_units_status
        source_units = (
            f" ({units} [{units_status.value}])"
            if units is not None and units_status is not MetadataStatus.UNKNOWN
            else " (units unknown)"
        )
        axis.set_ylabel(f"Source amplitude{source_units}")
        trace_axis = axis.twinx()
        if trace_indices.size:
            trace_axis.plot(
                trace_x,
                trace_values,
                color="tab:blue",
                linewidth=1.2,
                label="calcMUA trace",
            )
        trace_axis.set_ylabel(f"calcMUA ({output.mua_units})")
        axis.set_xlabel(
            "Time relative to original recording sample zero (s; no acquisition origin asserted)"
            if rate is not None
            else "Original recording sample index (0-based)"
        )
        axis.set_title(
            f"Channel {output.source_channel} · original calcMUA · {output.mode}"
        )
        handles_a, labels_a = axis.get_legend_handles_labels()
        handles_b, labels_b = trace_axis.get_legend_handles_labels()
        if handles_a or handles_b:
            axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
        interval = output.sample_interval
        filtering = (
            f"{output.filtering_description} [{output.filtering_status.value}]"
            if output.filtering_description
            else f"unavailable [{output.filtering_status.value}]"
        )
        source_status = "active" if source_indices.size else "inactive / unavailable"
        warning_text = " Warnings: " + " | ".join(record.warnings) if record.warnings else ""
        self.mua_display_label.setText(
            f"{record.operation_id} v{record.operation_version}; implementation "
            f"{record.implementation_version}; source SHA-256 {output.source_sha256}; "
            f"dataset {output.dataset_id}; {source_status} source; "
            f"original samples [{interval.start}, {interval.stop}); "
            f"{output.event_count} events; threshold {output.threshold:g} "
            f"{output.amplitude_units or '(units unknown)'} [{output.amplitude_units_status.value}]; "
            f"noise {output.noise_estimate:g}; "
            f"stdmin {output.stdmin:g}; bandwidth {output.bandwidth_s:g} s; "
            f"num_points {output.num_points} (does not change this explicit sample grid); "
            f"filtering {filtering}; execution {output.execution_id}. "
            "Trace points map to original samples; click to inspect without rerunning."
            f"{warning_text}"
        )
        self.mua_canvas.draw_idle()

    def _on_mua_plot_click(self, event) -> None:
        if self._mua_selection_views:
            if event.xdata is None or self.navigation_state is None:
                return
            first = self._mua_selection_views[0]
            rate = (
                float(first["sampling_rate_hz"])
                if first["sampling_rate_status"] is MetadataStatus.CONFIRMED
                else None
            )
            sample = round(event.xdata * rate) if rate else round(event.xdata)
            sample = min(max(sample, 0), self.navigation_state.sample_count - 1)
            width = self.navigation_state.visible_window.stop - self.navigation_state.visible_window.start
            start = min(max(sample - width // 2, 0), self.navigation_state.sample_count - width)
            self.navigation_state.zoom(start, start + width)
            self.workspace_tabs.setCurrentIndex(0)
            self._redraw_navigation_view()
            return
        if (
            event.xdata is None
            or self.active_registration is None
            or self.navigation_state is None
            or self._current_mua_record is None
            or not self._current_mua_record.data_references
            or self._current_mua_record.data_references[0].session_handle
            != self.active_registration.session_handle
        ):
            return
        output = self._current_mua_record.output
        sampling_rate = (
            output.sampling_rate_hz
            if isinstance(output, CalcMUAOutput)
            and output.sampling_rate_status is MetadataStatus.CONFIRMED
            else None
        )
        try:
            sample = sample_index_from_coordinate(
                float(event.xdata),
                sample_count=self.active_registration.sample_count,
                sampling_rate_hz=sampling_rate,
            )
        except (TypeError, ValueError):
            return
        margin = max(1, min(250, self.navigation_state.visible_window.stop - self.navigation_state.visible_window.start))
        start = max(0, sample - margin)
        stop = min(self.active_registration.sample_count, sample + margin + 1)
        self.workspace_tabs.setCurrentIndex(0)
        self._inspecting_event = True
        self.navigation_state.zoom(start, stop)
        self._redraw_navigation_view()
        self._inspecting_event = False

    def _sync_mua_navigation_limits(self, _event=None) -> None:
        """Mirror shared MUA time limits into the recording navigator."""
        if (
            self.navigation_state is None
            or not self._mua_selection_views
            or not self.mua_figure.axes
            or self.workspace_tabs.currentIndex() != 3
        ):
            return
        left, right = self.mua_figure.axes[0].get_xlim()
        first = self._mua_selection_views[0]
        rate = (
            float(first["sampling_rate_hz"])
            if first["sampling_rate_status"] is MetadataStatus.CONFIRMED
            else None
        )
        try:
            self.navigation_state.set_from_axis_limits(left, right, sampling_rate_hz=rate)
            self._redraw_navigation_view()
        except ValueError:
            return

    def _reload_isi_view(self, _value=None) -> None:
        record = self._current_isi_record
        if self.store is None or record is None or record.output is None:
            return
        output = record.output
        if not isinstance(output, ISIOutput):
            return
        group = self.isi_group_box.currentData()
        if group is None:
            self.isi_display_label.setText("No ISI group is available.")
            return
        channel, polarity = group
        artifact_root = self.store.isi_artifacts.root
        reference = output.interval_artifact_reference
        unit = output.interval_time_unit
        task = lambda: _read_isi_view(
            artifact_root,
            reference,
            int(channel),
            str(polarity),
            unit,
            self.isi_histogram_bins.value(),
            self.isi_page_start.value(),
        )
        self.isi_display_label.setText("Streaming interval artifact for histogram and selected rows…")
        self._start_view_worker("isi", task)

    def _render_isi_view(self) -> None:
        view = self._isi_artifact_view
        record = self._current_isi_record
        self.isi_figure.clear()
        axis = self.isi_figure.add_subplot(111)
        if view is None or record is None or record.output is None:
            axis.text(0.5, 0.5, "No ISI analysis selected.", ha="center", va="center", transform=axis.transAxes)
            self.isi_canvas.draw_idle()
            return
        output = record.output
        group = self.isi_group_box.currentData()
        summary = next(
            (item for item in output.summaries if (item.channel_index, item.polarity_group) == group),
            None,
        )
        edges = view["histogram_edges"]
        counts = view["histogram_counts"]
        unit = "ms" if output.interval_time_unit == "milliseconds" else "samples"
        if len(counts):
            axis.stairs(counts, edges, fill=True, alpha=0.45, color="tab:blue")
        else:
            axis.text(0.5, 0.5, "No intervals in this channel/polarity group.", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel(f"Inter-spike interval ({unit})")
        axis.set_ylabel("Interval count")
        axis.set_title(f"Channel {group[0]} · {group[1] if output.polarity_mode == 'separate' else 'combined polarity'}")
        self.isi_interval_table.setRowCount(len(view["page_rows"]))
        self._isi_page_rows = view["page_rows"]
        for row_index, item in enumerate(self._isi_page_rows):
            value = item["delta_ms"] if unit == "ms" else item["delta_samples"]
            value_text = f"{value:g} ms" if unit == "ms" else f"{value} samples"
            values = (
                item["starting_event_id"],
                item["ending_event_id"],
                item["channel_index"],
                item["polarity_group"],
                item["starting_sample_index"],
                item["ending_sample_index"],
                value_text,
            )
            for column, entry in enumerate(values):
                self.isi_interval_table.setItem(row_index, column, QTableWidgetItem(str(entry)))
        if summary is None:
            metrics = "No summary is available for this group."
        else:
            metrics = (
                f"Events {summary.detected_event_count}; intervals {summary.interval_count}; "
                f"mean {summary.mean_isi_ms if unit == 'ms' else summary.mean_isi_samples}; "
                f"median {summary.median_isi_ms if unit == 'ms' else summary.median_isi_samples}; "
                f"SD {summary.standard_deviation_isi_ms if unit == 'ms' else summary.standard_deviation_isi_samples}; "
                f"min {summary.minimum_isi_ms if unit == 'ms' else summary.minimum_isi_samples}; "
                f"max {summary.maximum_isi_ms if unit == 'ms' else summary.maximum_isi_samples}; "
                f"CV {summary.coefficient_of_variation}; duplicates {summary.duplicate_event_count}; "
                f"zero intervals {summary.zero_length_interval_count}; "
                f"below threshold {summary.intervals_below_threshold_count}; "
                f"short interval fraction {summary.short_interval_fraction}."
            )
        start = view["page_start"]
        total = view["interval_count"]
        end = min(start + len(view["page_rows"]), total)
        source_id = record.input_reference.dataset_id if record.input_reference else None
        stale = (
            self.active_registration is None
            or source_id != self.active_registration.dataset.dataset_id
        )
        prefix = "Historical result; source navigation is disabled. " if stale else ""
        threshold = (
            f"; explicit threshold {output.short_interval_threshold_ms} ms"
            if output.short_interval_threshold_ms is not None
            else f"; explicit threshold {output.short_interval_threshold_samples} samples"
            if output.short_interval_threshold_samples is not None
            else "; no short-interval threshold supplied"
        )
        self.isi_display_label.setText(
            f"{prefix}{metrics}{threshold}. Showing rows {start}–{end} of {total}. "
            "Short intervals are descriptive; they are not confirmed refractory violations."
        )
        self.isi_canvas.draw_idle()

    def _select_isi_interval(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(getattr(self, "_isi_page_rows", [])):
            return
        interval = self._isi_page_rows[row]
        record = self._current_isi_record
        output = record.output if record is not None else None
        detector_record = (
            self._detector_records.get(output.detector_execution_id)
            if isinstance(output, ISIOutput)
            else None
        )
        if detector_record is None or not detector_record.data_references:
            return
        if (
            self.active_registration is None
            or detector_record.data_references[0].dataset.dataset_id
            != self.active_registration.dataset.dataset_id
        ):
            self.isi_display_label.setText("This interval belongs to an inactive dataset.")
            return
        start_sample = int(interval["starting_sample_index"])
        end_sample = int(interval["ending_sample_index"])
        margin = max(1, min(250, self.active_registration.sample_count // 100 or 1))
        start = max(0, start_sample - margin)
        stop = min(self.active_registration.sample_count, end_sample + margin + 1)
        self.workspace_tabs.setCurrentIndex(0)
        self._inspecting_event = True
        self.navigation_state.zoom(start, stop)
        self._redraw_navigation_view()
        self._inspecting_event = False
        self.isi_display_label.setText(
            f"Interval {interval['starting_event_id']} → {interval['ending_event_id']}: "
            f"samples {start_sample}–{end_sample}; no events were modified."
        )

    def _load_waveform_record(self, record: ExecutionRecord) -> None:
        if self.store is None or record.output is None:
            return
        reference = record.data_references[0]
        if self.active_registration is None or reference.dataset.dataset_id != self.active_registration.dataset.dataset_id:
            self.waveform_display_label.setText(
                "Historical waveform result — displaying the saved artifact; source navigation is disabled."
            )
        output = record.output
        if not isinstance(output, WaveformExtractionOutput):
            self.waveform_display_label.setText("Waveform execution has an incompatible output type.")
            return
        self._waveform_artifact_view = None
        self.waveform_channel_box.blockSignals(True)
        self.waveform_channel_box.clear()
        nonempty = [item for item in output.channel_summaries if item.waveform_count]
        for summary in nonempty:
            self.waveform_channel_box.addItem(
                f"Channel {summary.channel_index} ({summary.waveform_count} waveforms)",
                summary.channel_index,
            )
        self.waveform_channel_box.blockSignals(False)
        if not nonempty:
            self.waveform_display_label.setText("Extraction contains no accepted waveforms.")
            self.waveform_figure.clear()
            self.waveform_canvas.draw_idle()
            return
        if self.waveform_channel_box.currentData() is None:
            self.waveform_channel_box.setCurrentIndex(0)
        self.workspace_tabs.setCurrentWidget(self.workspace_tabs.widget(1))
        self._request_waveform_view()

    def _populate_waveform_event_picker(self, view: dict[str, object]) -> None:
        events = view["accepted_events"]
        channel = self.waveform_channel_box.currentData()
        filtered = [event for event in events if channel is None or event["channel_index"] == channel]
        current = view.get("selected_event")
        current_id = current["event_id"] if current is not None else None
        self.waveform_event_box.blockSignals(True)
        self.waveform_event_box.clear()
        self.waveform_event_box.setEditable(True)
        for item in filtered[:250]:
            self.waveform_event_box.addItem(
                f"{item['event_id']} — sample {item['original_sample_index']}",
                item["event_id"],
            )
        if current_id is not None:
            found = self.waveform_event_box.findData(current_id)
            if found >= 0:
                self.waveform_event_box.setCurrentIndex(found)
            else:
                self.waveform_event_box.setEditText(str(current_id))
        elif filtered:
            self.waveform_event_box.setCurrentIndex(0)
        self.waveform_event_box.blockSignals(False)

    def _populate_waveform_excluded_table(self, view: dict[str, object]) -> None:
        excluded = view["excluded_events"]
        table = self.waveform_excluded_table
        table.setRowCount(min(len(excluded), 500))
        for row, item in enumerate(excluded[:500]):
            for column, value in enumerate(
                (item["event_id"], item["original_sample_index"], item["reason"])
            ):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        extra = max(0, len(excluded) - 500)
        if extra:
            self.waveform_display_label.setText(
                f"{extra} additional excluded events remain in the artifact mapping."
            )

    def _request_waveform_view(self, selected_event_id: str | None = None) -> None:
        record = self._current_waveform_record
        view = self._waveform_artifact_view
        if record is None or self.store is None or record.output is None:
            return
        channel = self.waveform_channel_box.currentData()
        if channel is None:
            return
        if selected_event_id is None and view is not None:
            selected = view.get("selected_event")
            selected_event_id = selected.get("event_id") if selected else None
        reference = record.output.artifact_reference
        artifact_root = self.store.artifacts.root
        maximum = self.waveform_max_display.value()
        task = lambda: _read_waveform_view(
            artifact_root,
            reference,
            int(channel),
            maximum,
            selected_event_id,
        )
        self.waveform_display_label.setText("Reading a bounded waveform subset…")
        self._start_view_worker("waveform", task)

    def _on_waveform_event_search(self) -> None:
        event_id = self.waveform_event_box.currentData() or self.waveform_event_box.currentText().split(" — ", 1)[0]
        if event_id:
            self._request_waveform_view(str(event_id))
            self._link_waveform_to_detection(str(event_id))

    def _on_waveform_view_selection_changed(self, _value=None) -> None:
        if self._waveform_artifact_view is not None:
            selected_id = self.waveform_event_box.currentData()
            if selected_id is None:
                selected_id = self.waveform_event_box.currentText().split(" — ", 1)[0]
            candidates = [
                event for event in self._waveform_artifact_view["channel_events"]
                if event["channel_index"] == self.waveform_channel_box.currentData()
            ]
            if not selected_id or not any(event["event_id"] == selected_id for event in candidates):
                selected_id = candidates[0]["event_id"] if candidates else None
            self._request_waveform_view(str(selected_id) if selected_id is not None else None)

    def _on_waveform_event_selection_changed(self, _index: int) -> None:
        if self._waveform_artifact_view is None:
            return
        selected_id = self.waveform_event_box.currentData()
        if selected_id is None:
            selected_id = self.waveform_event_box.currentText().split(" — ", 1)[0]
        if selected_id:
            self._request_waveform_view(str(selected_id))
            self._link_waveform_to_detection(str(selected_id))

    def _render_waveform_view(self, *_args) -> None:
        view = self._waveform_artifact_view
        record = self._current_waveform_record
        self.waveform_figure.clear()
        axis = self.waveform_figure.add_subplot(111)
        if view is None or record is None or record.output is None:
            axis.text(0.5, 0.5, "No waveform extraction selected.", ha="center", va="center", transform=axis.transAxes)
            self.waveform_canvas.draw_idle()
            return
        manifest = view["manifest"]
        channel = int(view["channel_index"])
        offsets = np.arange(int(manifest["waveform_shape"][1])) + int(
            manifest["sample_offsets"]["start_inclusive"]
        )
        rate_status = manifest.get("sampling_rate_status", "unknown")
        rate = manifest.get("sampling_rate_hz")
        x_values = offsets / float(rate) if rate_status == "confirmed" and rate else offsets
        summaries = view["summaries"]
        display_events = view["display_events"]
        display_waveforms = view["display_waveforms"]
        selected = view.get("selected_event")
        selected_id = selected.get("event_id") if selected else None
        selected_is_displayed = False
        if self.waveform_show_individual.isChecked():
            for event, waveform in zip(display_events, display_waveforms, strict=True):
                if event["event_id"] == selected_id:
                    selected_is_displayed = True
                    continue
                axis.plot(x_values, waveform, color="0.35", alpha=0.25, linewidth=0.6)
        if selected is not None and not selected_is_displayed and view["selected_waveform"] is not None:
            axis.plot(x_values, view["selected_waveform"], color="tab:red", linewidth=1.4, label="Selected event")
        mean = summaries.get("mean")
        standard_deviation = summaries.get("standard-deviation")
        if self.waveform_show_std.isChecked() and mean is not None and standard_deviation is not None:
            axis.fill_between(
                x_values,
                mean - standard_deviation,
                mean + standard_deviation,
                color="tab:blue",
                alpha=0.2,
                label="Mean ± population SD",
            )
        if self.waveform_show_mean.isChecked() and mean is not None:
            axis.plot(x_values, mean, color="tab:blue", linewidth=1.5, label="Mean")
        if self.waveform_show_median.isChecked() and summaries.get("median") is not None:
            axis.plot(x_values, summaries["median"], color="tab:green", linewidth=1.3, label="Median")
        if selected is not None and view["selected_waveform"] is not None and selected_is_displayed:
            axis.plot(x_values, view["selected_waveform"], color="tab:red", linewidth=1.4, label="Selected event")
        if rate_status == "confirmed" and rate:
            axis.set_xlabel("Time relative to detector event (s)")
        else:
            axis.set_xlabel("Sample offset from detector event")
        units = manifest.get("amplitude_units")
        units_status = manifest.get("amplitude_units_status", "unknown")
        axis.set_ylabel(
            f"Amplitude ({units})" if units is not None and units_status != "unknown" else "Amplitude (units unknown)"
        )
        axis.set_title(f"Channel {channel} — detector-aligned waveforms")
        if axis.lines or axis.collections:
            axis.legend(loc="best")
        channel_events = view["channel_events"]
        displayed_count = len(display_events)
        stale = record.data_references[0].session_handle != getattr(self.active_registration, "session_handle", None)
        status = "Historical/stale — " if stale else ""
        self.waveform_display_label.setText(
            f"{status}{len(channel_events)} accepted waveform(s), {len(view['excluded_events'])} boundary-excluded; "
            f"displaying {displayed_count} individual trace(s) of {len(channel_events)} (deterministic subset). "
            "Subset affects visualization only."
        )
        self.waveform_canvas.draw_idle()

    def _highlight_waveform_event(self, event: dict[str, object], detector_record: ExecutionRecord) -> None:
        waveform_record = self._current_waveform_record
        if waveform_record is None or waveform_record.output is None:
            return
        if waveform_record.output.detection_reference != f"execution://{detector_record.execution_id}":
            return
        event_id = str(event["event_id"])
        self.workspace_tabs.setCurrentIndex(1)
        self.waveform_event_box.setEditText(event_id)
        self._request_waveform_view(event_id)

    def _link_waveform_to_detection(self, event_id: str) -> None:
        waveform_record = self._current_waveform_record
        if waveform_record is None or waveform_record.output is None:
            return
        execution_id = waveform_record.output.detection_reference.removeprefix("execution://")
        detector_record = self._detector_records.get(execution_id)
        if detector_record is None or detector_record.status is not ExecutionStatus.SUCCEEDED:
            return
        if (
            self.active_registration is None
            or not detector_record.data_references
            or detector_record.data_references[0].dataset.dataset_id
            != self.active_registration.dataset.dataset_id
        ):
            return
        if self._current_record is not detector_record:
            self._current_record = detector_record
            self._current_events = self._events_from_record(detector_record)
            self._result_stale = False
            self._populate_event_table()
            self._refresh_detection_artists()
        row = next(
            (index for index, item in enumerate(self._current_events) if item["event_id"] == event_id),
            None,
        )
        if row is not None:
            self.event_table.selectRow(row)
            self._linking_event = True
            try:
                self._select_detected_event(row, 0)
            finally:
                self._linking_event = False

    def approve_and_run(self) -> None:
        """Issue approval for the displayed request, then execute it on a worker."""
        if (
            self._prepared_execution is None
            and self._prepared_mua_batch is None
        ) or self._execution_in_flight:
            return
        if self.store is None:
            QMessageBox.information(
                self,
                "Project required",
                "Select or create a project first. Then review and approve the operation.",
            )
            return
        confirmation = QMessageBox.question(
            self,
            "Approve scientific operation",
            (
                "Approve and run every independently prepared channel operation listed in the review panel?"
                if self._prepared_mua_batch is not None
                else "Approve and run exactly the prepared operation shown in the review panel?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        approved_by, accepted = QInputDialog.getText(
            self,
            "Approval provenance",
            "Researcher name or identifier to record with this approval:",
        )
        if not accepted or not approved_by.strip():
            return
        try:
            if self._prepared_mua_batch is not None:
                approvals = tuple(
                    self.operation_runner.approve(item, approved_by=approved_by.strip())
                    for item in self._prepared_mua_batch
                )
            else:
                approvals = (self.operation_runner.approve(
                    self._prepared_execution, approved_by=approved_by.strip()
                ),)
        except (TypeError, ValueError) as exc:
            self._invalidate_prepared(f"approval failed: {exc}")
            QMessageBox.warning(self, "Approval unavailable", str(exc))
            return

        if self._assistant_approval_proposal_id is not None:
            proposal_id = self._assistant_approval_proposal_id
            self._assistant_tools.dismiss_proposal(proposal_id)
            self._assistant_approval_proposal_id = None
            self._assistant_approval_ids.update(item.approval_id for item in approvals)
            selected = self.assistant_proposal_list.currentItem()
            if selected is not None and selected.data(Qt.UserRole) == proposal_id:
                self.assistant_proposal_list.takeItem(
                    self.assistant_proposal_list.row(selected)
                )
            self.assistant_proposal_details.clear()
            self.assistant_activity_label.setText(
                "Researcher approved the proposal; operation is running."
            )

        # A prepared approval is single-use in this interface. Re-running requires
        # a fresh review and explicit approval, preventing accidental duplicates.
        self._prepared_execution = None
        self._prepared_mua_batch = None
        self._execution_in_flight = True
        thread = QThread(self)
        worker = (
            _OperationBatchWorker(self.operation_runner, approvals, self._dataset_generation)
            if len(approvals) > 1
            else _OperationWorker(self.operation_runner, approvals[0], self._dataset_generation)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_operation_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_worker_thread_finished)
        self._worker_thread = thread
        self._worker = worker
        self.analysis_state_label.setText("Running approved operation…")
        self._update_action_state()
        thread.start()

    def _on_operation_finished(
        self,
        dataset_generation: int,
        record: ExecutionRecord | None,
        error: Exception | None,
    ) -> None:
        """Consume worker results on the GUI thread and persist typed records."""
        if isinstance(record, list):
            batch_records = tuple(item for item in record if isinstance(item, ExecutionRecord))
            for item in batch_records:
                self._on_operation_finished(dataset_generation, item, None)
            successful = sum(item.status is ExecutionStatus.SUCCEEDED for item in batch_records)
            attempted = len(batch_records)
            if attempted:
                self.operation_review.append(
                    f"\nMultichannel calcMUA batch: {successful}/{attempted} channel operations succeeded."
                )
                self.analysis_state_label.setText(
                    f"{'Completed' if successful == attempted else 'Partial failure'} — "
                    f"{successful}/{attempted} channel calcMUA result(s) saved."
                )
                self._on_mua_display_changed()
            if error is not None:
                self.operation_review.append(f"\nBatch stopped after partial execution: {error}")
                self.analysis_state_label.setText(
                    f"Partial failure — {attempted} channel operation(s) completed before {error}."
                )
            return
        if error is not None or record is None:
            self.analysis_state_label.setText(f"Failed — {error}")
            if self._assistant_approval_ids:
                self.chat_history.append(
                    f"<i>Approved assistant proposal failed before producing a record: {html.escape(str(error))}</i>"
                )
                self._assistant_approval_ids.clear()
            QMessageBox.critical(self, "Analysis failed", str(error))
            return
        self._execution_records.append(record)
        if record.provenance.approval_id in self._assistant_approval_ids:
            self._assistant_approval_ids.discard(record.provenance.approval_id)
            self.chat_history.append(
                f"<i>Approved proposal outcome: {html.escape(record.operation_id)} "
                f"{html.escape(record.status.value)}; execution {html.escape(record.execution_id)}.</i>"
            )
        is_current_dataset = dataset_generation == self._dataset_generation
        if record.operation_id in {NEGATIVE_DETECTOR_ID, SIGNED_PEAK_ID}:
            if record.status is ExecutionStatus.SUCCEEDED:
                self._detector_records[record.execution_id] = record
                self._current_record = record
                self._current_events = self._events_from_record(record)
                self._result_stale = not is_current_dataset
                self._populate_event_table()
                self._refresh_detector_execution_choices()
                count = len(self._current_events)
                state = f"Completed — {count} event(s)"
                if self._result_stale:
                    state = "Stale configuration — completed result retained; dataset changed."
                self.analysis_state_label.setText(state)
                self._refresh_detection_artists()
            else:
                self._current_record = record
                self._current_events = []
                self._result_stale = not is_current_dataset
                self._populate_event_table()
                message = record.error.message if record.error is not None else "unknown detector error"
                self.analysis_state_label.setText(f"Failed — {message}")
        elif record.operation_id == "neuroscope.spike.extract-waveforms":
            if record.status is ExecutionStatus.SUCCEEDED and is_current_dataset:
                self._current_waveform_record = record
                self.waveform_display_label.setText("Waveform extraction completed; opening artifact view…")
                self._load_waveform_record(record)
            elif record.status is ExecutionStatus.SUCCEEDED:
                self._current_waveform_record = record
                self.waveform_display_label.setText("Stale waveform result retained; its dataset changed during execution.")
            else:
                message = record.error.message if record.error is not None else "unknown waveform error"
                self.waveform_display_label.setText(f"Waveform extraction failed: {message}")
        elif record.operation_id == "neuroscope.spike.analyze-isi":
            if record.status is ExecutionStatus.SUCCEEDED and is_current_dataset:
                self._current_isi_record = record
                self.isi_display_label.setText("ISI analysis completed; opening interval artifact…")
                self._load_isi_record(record)
            elif record.status is ExecutionStatus.SUCCEEDED:
                self._current_isi_record = record
                self.isi_display_label.setText("Stale ISI result retained; its dataset changed during execution.")
            else:
                message = record.error.message if record.error is not None else "unknown ISI error"
                self.isi_display_label.setText(f"ISI analysis failed: {message}")
        elif record.operation_id == CALC_MUA_ID:
            if record.status is ExecutionStatus.SUCCEEDED:
                self._register_mua_result(record, average=False)
            if record.status is ExecutionStatus.SUCCEEDED and is_current_dataset:
                self._current_mua_record = record
                self.mua_display_label.setText("calcMUA completed; opening project artifact…")
                self._load_mua_record(record)
            elif record.status is ExecutionStatus.SUCCEEDED:
                self._current_mua_record = record
                self.mua_display_label.setText(
                    "Stale calcMUA result retained; the dataset changed during execution."
                )
            else:
                message = record.error.message if record.error is not None else "unknown calcMUA error"
                self.mua_display_label.setText(f"calcMUA failed: {message}")
        elif record.operation_id == AVERAGE_MUA_OPERATION.operation_id:
            if record.status is ExecutionStatus.SUCCEEDED:
                self._register_mua_result(record, average=True)
                self._current_mua_record = None
                self.mua_display_label.setText("Averaged MUAtrace artifact created; opening it for display…")
                self._on_mua_display_changed()
            else:
                message = record.error.message if record.error is not None else "unknown averaging error"
                self.mua_display_label.setText(f"MUAtrace averaging failed: {message}")
        self.operation_review.append(
            f"\nExecution {record.execution_id} ({record.operation_id}) {record.status.value} at {record.finished_at_utc.isoformat()} UTC."
        )
        if record.error is not None:
            self.operation_review.append(f"Execution error: {record.error.message}")

        try:
            self.store.append_execution_record(record)
            self._pending_save_record = None
            self.save_record_button.setVisible(False)
            self.operation_review.append("Execution record and approval provenance saved to project history.")
        except (OSError, TypeError, ValueError) as exc:
            self._pending_save_record = record
            self.save_record_button.setVisible(True)
            self.operation_review.append(
                f"Execution record is not yet saved. Save failed: {exc}"
            )
        self._refresh_detection_artists()
        self.canvas.draw_idle()

    def _on_worker_thread_finished(self) -> None:
        thread = self._worker_thread
        self._worker_thread = None
        self._worker = None
        self._execution_in_flight = False
        if thread is not None:
            thread.deleteLater()
        self._update_action_state()

    def retry_save_execution_record(self) -> None:
        record = self._pending_save_record
        if record is None or self.store is None:
            return
        try:
            self.store.append_execution_record(record)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save record", str(exc))
            return
        self._pending_save_record = None
        self.save_record_button.setVisible(False)
        self.operation_review.append("Execution record saved to project history.")

    def _events_from_record(self, record: ExecutionRecord) -> list[dict[str, object]]:
        output = record.output
        if output is None:
            return []
        indices = tuple(output.original_sample_indices)
        relative_indices = tuple(output.window_relative_sample_indices)
        amplitudes = tuple(output.amplitudes)
        channel = int(output.channel_index)
        polarity = (
            "negative"
            if record.operation_id == NEGATIVE_DETECTOR_ID
            else str(output.polarity)
        )
        relative_times = getattr(output, "times_s_relative_to_window", None)
        time_values = tuple(relative_times) if relative_times is not None else (None,) * len(indices)
        return [
            {
                "event_id": f"{record.execution_id}:{position}",
                "row_reference": f"execution://{record.execution_id}#{position}",
                "original_sample_index": int(sample),
                "window_relative_sample_index": int(relative),
                "channel_index": channel,
                "amplitude": float(amplitude),
                "polarity": polarity,
                "time_seconds_relative_to_window": (
                    float(time_value) if time_value is not None else None
                ),
                "units": getattr(output, "amplitude_units", None),
                "units_status": getattr(output, "amplitude_units_status", None),
            }
            for position, (sample, relative, amplitude, time_value) in enumerate(
                zip(indices, relative_indices, amplitudes, time_values, strict=True)
            )
        ]

    def _populate_event_table(self) -> None:
        self.event_table.blockSignals(True)
        self.event_table.setRowCount(len(self._current_events))
        for row, item in enumerate(self._current_events):
            units = item["units"]
            unit_status = item["units_status"]
            amplitude_label = f"{item['amplitude']:g}"
            if units is not None and unit_status is not MetadataStatus.UNKNOWN:
                amplitude_label += f" {units} ({unit_status.value})"
            else:
                amplitude_label += " (units unknown)"
            values = (
                str(item["event_id"]),
                str(item["original_sample_index"]),
                str(item["channel_index"]),
                str(item["polarity"]),
                amplitude_label,
            )
            for column, value in enumerate(values):
                self.event_table.setItem(row, column, QTableWidgetItem(value))
        self.event_table.blockSignals(False)
        if self._current_record is not None and not self._current_events:
            self.operation_review.append("\nNo events detected in the selected interval.")

    def _select_detected_event(self, row: int, _column: int) -> None:
        if not 0 <= row < len(self._current_events) or self._current_record is None:
            return
        item = self._current_events[row]
        record = self._current_record
        output = record.output
        rate = getattr(output, "sampling_rate_hz", None) if output is not None else None
        rate_status = getattr(output, "sampling_rate_status", None) if output is not None else None
        sample = int(item["original_sample_index"])
        details = [
            f"Event reference: {item['row_reference']}",
            f"Original sample index: {sample}",
            f"Detector-window sample index: {item['window_relative_sample_index']}",
            f"Channel (0-based): {item['channel_index']}",
            f"Signed amplitude: {item['amplitude']:g} {item['units'] or '(units unknown)'}",
            f"Polarity: {item['polarity']}",
            f"Detector: {record.operation_id} v{record.operation_version}",
            f"Execution reference: execution://{record.execution_id}",
        ]
        if rate_status is MetadataStatus.CONFIRMED and rate is not None:
            details.append(
                f"Time relative to recording sample 0: {sample / float(rate):.9g} s (no acquisition origin asserted)"
            )
            local_time = item["time_seconds_relative_to_window"]
            if local_time is not None:
                details.append(f"Time relative to selected detector window: {float(local_time):.9g} s")
        else:
            details.append("Time in seconds unavailable: sampling rate is not confirmed.")
        self.event_details.setPlainText("\n".join(details))
        if not self._linking_event:
            self._highlight_waveform_event(item, record)

        registration = self.active_registration
        if registration is None or not record.data_references:
            return
        analyzed_reference = record.data_references[0]
        if analyzed_reference.dataset.dataset_id != registration.dataset.dataset_id:
            self.event_details.append("\nThis historical event belongs to a different dataset identity; source navigation is disabled.")
            return
        if not 0 <= sample < registration.sample_count:
            return
        if self.navigation_state is not None:
            margin = max(1, min(250, self.navigation_state.visible_window.stop - self.navigation_state.visible_window.start))
            start = max(0, sample - margin)
            stop = min(registration.sample_count, sample + margin + 1)
            if stop <= start:
                return
            self._inspecting_event = True
            self.navigation_state.zoom(start, stop)
            self._redraw_navigation_view()
            self._inspecting_event = False

    def _refresh_detection_artists(self) -> None:
        """Overlay only active-dataset detections inside the current visible window."""
        if not self._plot_channel_by_axis:
            return
        record = self._current_record
        registration = self.active_registration
        navigation = self.navigation_state
        if (
            record is None
            or registration is None
            or navigation is None
            or not record.data_references
            or record.data_references[0].dataset.dataset_id != registration.dataset.dataset_id
        ):
            return
        rate = self.active_dataset.exploration.summary.metadata["sampling_rate_hz"]
        confirmed_rate = (
            float(rate.value)
            if rate.status is MetadataStatus.CONFIRMED and rate.value is not None
            else None
        )
        window = navigation.visible_window
        for axis, channel in self._plot_channel_by_axis.items():
            scatter = self._event_scatters.get(axis)
            if scatter is None:
                scatter = axis.scatter([], [], s=20, marker="v", zorder=4)
                self._event_scatters[axis] = scatter
            channel_events = [
                item for item in self._current_events
                if item["channel_index"] == channel
                and window.start <= item["original_sample_index"] < window.stop
            ]
            x_values = [
                item["original_sample_index"] / confirmed_rate
                if confirmed_rate is not None
                else item["original_sample_index"]
                for item in channel_events
            ]
            y_values = [item["amplitude"] for item in channel_events]
            scatter.set_offsets(np.column_stack((x_values, y_values)))
            scatter.set_facecolor("0.5" if self._result_stale else "tab:red")
            scatter.set_edgecolor("black")
            scatter.set_visible(bool(channel_events))

    def _assistant_state(self) -> AssistantApplicationState:
        """Take a metadata-only view of the current app context for local tools."""
        current_reference = None
        if self.active_registration is not None and self.navigation_state is not None:
            selected = self.navigation_state.selected_channels
            window = self.navigation_state.visible_window
            if selected and window.stop > window.start:
                current_reference = self.active_registration.reference(
                    selected, SampleInterval(start=window.start, stop=window.stop)
                )
        return AssistantApplicationState(
            profile=(self.active_dataset.profile if self.active_dataset is not None else None),
            registration=self.active_registration,
            data_store=self.dataset_store,
            registry=self.operation_registry,
            runner=self.operation_runner,
            project_store=self.store,
            generation=self._assistant_generation,
            current_data_reference=current_reference,
            literature_store=self._effective_literature_store(),
            active_literature_library_ids=(
                self._effective_literature_store().active_library_ids()
                if self._effective_literature_store() is not None else ()
            ),
            external_provider=(
                self._assistant_provider_kind == "openai"
                or isinstance(self._assistant_provider, OllamaProvider) and self._assistant_provider.is_remote
            ),
            externally_shareable_document_ids=self._assistant_external_document_allowlist,
        )

    def _effective_literature_store(self) -> LiteratureLibraryStore | None:
        if self._literature_store is not None:
            return self._literature_store
        return self.store.literature if self.store is not None else None

    def _current_literature_library_id(self) -> str | None:
        if self._literature_store is None or not hasattr(self, "literature_library_box"):
            return None
        item = self.literature_library_box.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _selected_literature_library_ids(self) -> tuple[str, ...]:
        if self._literature_store is None:
            return ()
        return self._literature_store.active_library_ids()

    def _refresh_literature_libraries(self) -> None:
        if not hasattr(self, "literature_library_box"):
            return
        box = self.literature_library_box
        selected = self._current_literature_library_id()
        box.blockSignals(True)
        box.clear()
        if self._literature_store is not None:
            try:
                active = set(self._literature_store.active_library_ids())
            except ValueError as exc:
                active = set()
                self.literature_results.setPlainText(
                    f"Saved active corpus selection is invalid. Choose libraries below to replace it: {exc}"
                )
            try:
                libraries = self._literature_store.list_libraries()
            except ValueError as exc:
                libraries = ()
                self.literature_results.setPlainText(f"Library catalog is damaged or unavailable: {exc}")
            for library in libraries:
                pack_label = (
                    f" · Knowledge Pack {library.knowledge_pack_version}"
                    if library.knowledge_pack_id else ""
                )
                item = QListWidgetItem(f"{library.name} · {library.scope}{pack_label}")
                item.setData(Qt.UserRole, str(library.library_id))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked if str(library.library_id) in active else Qt.Unchecked)
                box.addItem(item)
            if selected:
                index = next((i for i in range(box.count()) if box.item(i).data(Qt.UserRole) == selected), -1)
                if index >= 0:
                    box.setCurrentRow(index)
            if box.currentRow() < 0 and box.count():
                box.setCurrentRow(0)
        box.blockSignals(False)
        self._refresh_literature_documents()

    def _refresh_literature_documents(self) -> None:
        table = self.literature_documents
        table.blockSignals(True)
        table.setRowCount(0)
        library_id = self._current_literature_library_id()
        if self._literature_store is not None and library_id:
            try:
                documents = self._literature_store.list_documents(library_id)
                integrity = {
                    entry["document_id"]: entry["status"]
                    for entry in self._literature_store.validate_existing()
                }
                for document in documents:
                    row = table.rowCount(); table.insertRow(row)
                    visible = QTableWidgetItem("")
                    visible.setFlags(visible.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    visible.setCheckState(Qt.Checked if document.assistant_visible else Qt.Unchecked)
                    visible.setData(Qt.UserRole, str(document.document_id))
                    table.setItem(row, 0, visible)
                    for column, value in enumerate((
                        document.display_title,
                        document.metadata.doi or "Unknown",
                        document.metadata.publication_status,
                        document.metadata.access_status,
                        integrity.get(str(document.document_id), document.extraction_status),
                    ), start=1):
                        cell = QTableWidgetItem(value)
                        cell.setData(Qt.UserRole, str(document.document_id))
                        table.setItem(row, column, cell)
            except (OSError, ValueError) as exc:
                self.literature_results.setPlainText(f"Library could not be opened: {exc}")
        table.blockSignals(False)
        table.resizeColumnsToContents()

    def _on_literature_library_selection_changed(self, _item=None) -> None:
        if self._literature_store is None:
            return
        try:
            active = tuple(
                self.literature_library_box.item(i).data(Qt.UserRole)
                for i in range(self.literature_library_box.count())
                if self.literature_library_box.item(i).checkState() == Qt.Checked
            )
            self._literature_store.set_active_libraries(active)
            self._refresh_literature_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Library selection unavailable", str(exc))

    def _on_literature_document_visibility_changed(self, item) -> None:
        if self._literature_store is None or item.column() != 0:
            return
        library_id = self._current_literature_library_id()
        document_id = item.data(Qt.UserRole)
        if not library_id or not document_id:
            return
        try:
            self._literature_store.set_assistant_visibility(
                library_id, document_id, item.checkState() == Qt.Checked
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Document visibility unavailable", str(exc))

    def _open_literature_catalog(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open or create a separate local literature catalog", ""
        )
        if not directory:
            return
        try:
            candidate = LiteratureLibraryStore(directory, create=True)
            candidate.validate_existing()
            self._literature_store = candidate
            self._refresh_literature_libraries()
            self.literature_results.setPlainText(
                "Opened the explicitly selected local catalog. It is active only in this application context; "
                "opening another project resets the catalog selection to that project's own library."
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Local catalog unavailable", str(exc))

    def _create_literature_library(self) -> None:
        if self._literature_store is None:
            QMessageBox.information(self, "Project required", "Select or create a project before creating its local literature libraries.")
            return
        try:
            self._literature_store.create_library(
                self.literature_name_input.text(), self.literature_scope_box.currentData()
            )
            self.literature_name_input.clear()
            self._refresh_literature_libraries()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Library could not be created", str(exc))

    def _import_literature_documents(self) -> None:
        library_id = self._current_literature_library_id()
        if self._literature_store is None or not library_id:
            QMessageBox.information(self, "Select a library", "Create or select a library first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import authorized scientific documents", "",
            "Scientific documents (*.pdf *.txt *.md *.markdown)"
        )
        if not paths:
            return
        retain = QMessageBox.question(
            self, "Retain source documents?",
            "Copy these explicitly selected source files into this project library for portable reopening? This may duplicate copyrighted or confidential contents. Choose No to store extracted text and metadata only.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No,
        )
        if retain == QMessageBox.Cancel:
            return
        if self._literature_import_thread is not None and self._literature_import_thread.isRunning():
            self.literature_results.setPlainText("A literature import is already running.")
            return
        self.literature_import_button.setEnabled(False)
        self.literature_results.setPlainText(f"Extracting {len(paths)} local document(s)…")
        thread = QThread(self)
        worker = _LiteratureImportWorker(
            self._literature_store, library_id, paths, retain == QMessageBox.Yes,
            self._assistant_generation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_literature_import_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_literature_import_thread_finished)
        self._literature_import_thread = thread
        self._literature_import_worker = worker
        thread.start()

    @Slot(object, object, int, object)
    def _on_literature_import_finished(self, store, library_id, generation: int, result) -> None:
        if store is not self._literature_store or generation != self._assistant_generation:
            return
        imported, failures = result
        self._refresh_literature_documents()
        self.literature_results.setPlainText(
            f"Imported {imported} document(s).\n" + "\n".join(failures)
        )

    @Slot()
    def _on_literature_import_thread_finished(self) -> None:
        thread = self._literature_import_thread
        self._literature_import_thread = None
        self._literature_import_worker = None
        self.literature_import_button.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def _remove_literature_document(self) -> None:
        if self._literature_store is None or self.literature_documents.currentRow() < 0:
            return
        item = self.literature_documents.item(self.literature_documents.currentRow(), 0)
        document_id = item.data(Qt.UserRole) if item else None
        library_id = self._current_literature_library_id()
        if not document_id or not library_id:
            return
        answer = QMessageBox.question(
            self, "Remove document", "Remove the document, extracted passages, and any source copy retained by this library? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self._literature_store.remove_document(library_id, document_id)
            self._refresh_literature_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Document could not be removed", str(exc))

    def _inspect_literature_document(self) -> None:
        if self._literature_store is None or self.literature_documents.currentRow() < 0:
            return
        row = self.literature_documents.currentRow()
        item = self.literature_documents.item(row, 0)
        library_id = self._current_literature_library_id()
        try:
            document = self._literature_store.get_document(library_id, item.data(Qt.UserRole))
            metadata = document.metadata.model_dump(mode="json")
            details = {
                "document_id": str(document.document_id),
                "library_id": str(document.library_id),
                "title": document.display_title,
                "source_filename": document.source_filename,
                "metadata": metadata,
                "content_sha256": document.content_sha256,
                "imported_at_utc": document.imported_at_utc.isoformat(),
                "extraction_status": document.extraction_status,
                "extraction_warnings": document.extraction_warnings,
                "index_version": document.index_version,
                "assistant_visible": document.assistant_visible,
                "source_retained": document.source_retained,
                "provenance": document.provenance,
            }
            QMessageBox.information(
                self, "Literature document metadata",
                json.dumps(details, ensure_ascii=False, indent=2),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Metadata unavailable", str(exc))

    def _edit_literature_status(self) -> None:
        if self._literature_store is None or self.literature_documents.currentRow() < 0:
            return
        library_id = self._current_literature_library_id()
        item = self.literature_documents.item(self.literature_documents.currentRow(), 0)
        if not library_id or item is None:
            return
        try:
            document = self._literature_store.get_document(library_id, item.data(Qt.UserRole))
            access_values = ["unknown", "public", "private"]
            status_values = ["unknown", "unverified", "preprint", "peer_reviewed"]
            access, accepted = QInputDialog.getItem(
                self, "Set explicit access status",
                "Choose only a status supported by information you are authorized to provide:",
                access_values, access_values.index(document.metadata.access_status), False,
            )
            if not accepted:
                return
            status, accepted = QInputDialog.getItem(
                self, "Set publication/review status",
                "A DOI does not verify peer review. A manually supplied peer-reviewed status is labeled researcher-entered.",
                status_values, status_values.index(document.metadata.publication_status), False,
            )
            if not accepted:
                return
            metadata = document.metadata.model_copy(update={
                "access_status": access,
                "publication_status": status,
                "publication_status_source": (
                    "researcher_entered_not_independently_verified"
                    if status == "peer_reviewed" else None
                ),
            })
            self._literature_store.update_document_metadata(
                library_id, document.document_id, metadata,
                provenance_note="researcher_explicit_status_selection",
            )
            self._refresh_literature_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Metadata update failed", str(exc))

    def _replace_literature_document(self) -> None:
        if self._literature_store is None or self.literature_documents.currentRow() < 0:
            return
        library_id = self._current_literature_library_id()
        selected = self.literature_documents.item(self.literature_documents.currentRow(), 0)
        if not library_id or selected is None:
            return
        document_id = selected.data(Qt.UserRole)
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose replacement source", "",
            "Scientific documents (*.pdf *.txt *.md *.markdown)",
        )
        if not path:
            return
        if QMessageBox.question(
            self, "Confirm document replacement",
            "Replace the content for this stable document ID and rebuild its passage index? The old content fingerprint will no longer be active.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        retain = QMessageBox.question(
            self, "Retain replacement source?",
            "Copy the selected replacement source into this library?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No,
        )
        if retain == QMessageBox.Cancel:
            return
        try:
            previous = self._literature_store.get_document(library_id, document_id)
            self._literature_store.import_document(
                library_id, path,
                metadata=previous.metadata,
                assistant_visible=previous.assistant_visible,
                retain_original=retain == QMessageBox.Yes,
                document_id=previous.document_id,
                replace_existing=True,
            )
            self._refresh_literature_documents()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Replacement failed", str(exc))

    def _search_literature_library(self) -> None:
        if self._literature_store is None:
            self.literature_results.setPlainText("Select a project and explicitly select active libraries before searching.")
            return
        try:
            hits = self._literature_store.search(
                self.literature_query_input.text(),
                selected_library_ids=self._literature_store.active_library_ids(),
                limit=10,
            )
            blocks = []
            for hit in hits:
                warning = "\nPreprint — not peer reviewed." if hit.document.metadata.publication_status == "preprint" else ""
                location = f"page {hit.passage.page_number}" if hit.passage.page_number else f"characters {hit.passage.char_start}:{hit.passage.char_stop}"
                blocks.append(
                    f"{hit.document.display_title} · {hit.library_name} ({hit.library_id})\n"
                    f"Document {hit.document.document_id} · content SHA-256 {hit.document.content_sha256} · passage {hit.passage.passage_id} · {location}\n"
                    f"Knowledge Pack: {hit.knowledge_pack_id or 'none'} · version {hit.knowledge_pack_version or 'n/a'}\n"
                    f"DOI: {hit.document.metadata.doi or 'Unknown'} · status: {hit.document.metadata.publication_status} · access: {hit.document.metadata.access_status}{warning}\n"
                    f"Relevance score {hit.score:.4f} (not scientific confidence)\n\n{hit.passage.text}"
                )
            self.literature_results.setPlainText("\n\n---\n\n".join(blocks) if blocks else "No matching indexed passages were found in the selected libraries.")
        except (OSError, ValueError) as exc:
            self.literature_results.setPlainText(f"Search unavailable: {exc}")

    def _export_literature_manifest(self) -> None:
        library_id = self._current_literature_library_id()
        if self._literature_store is None or not library_id:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export metadata-only library manifest", "library-manifest.json", "JSON (*.json)")
        if not path:
            return
        try:
            self._literature_store.export_manifest(library_id, path)
            self.literature_results.setPlainText("Metadata-only manifest exported. Document text and source files were not included.")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Manifest export failed", str(exc))

    def _export_literature_payloads(self) -> None:
        library_id = self._current_literature_library_id()
        if self._literature_store is None or not library_id:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export retained document payloads", "literature-library.zip", "ZIP (*.zip)"
        )
        if not path:
            return
        answer = QMessageBox.question(
            self, "Confirm document-content export",
            "This export may include retained copyrighted or confidential source documents. Confirm that you are authorized to export these document payloads. Metadata-only manifest export does not include contents.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self._literature_store.export_payload_bundle(
                library_id, path, researcher_confirmed=True
            )
            self.literature_results.setPlainText("Explicitly authorized payload bundle exported. Existing files were not overwritten.")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Payload export failed", str(exc))

    def _import_literature_manifest(self) -> None:
        if self._literature_store is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import compatible metadata-only library manifest", "", "JSON (*.json)")
        if not path:
            return
        try:
            self._literature_store.import_manifest(path)
            self._refresh_literature_libraries()
            self.literature_results.setPlainText("Manifest imported. Document payloads and extracted passages are intentionally absent; source documents must be explicitly re-imported before search.")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Manifest import failed", str(exc))

    def _create_knowledge_pack(self) -> None:
        if self._literature_store is None:
            return
        library_id = self._current_literature_library_id()
        if not library_id:
            QMessageBox.information(self, "Knowledge Pack", "Select a source library first.")
            return
        selected_rows = self.literature_documents.selectionModel().selectedRows()
        document_ids = tuple(
            self.literature_documents.item(index.row(), 0).data(Qt.UserRole)
            for index in selected_rows
            if self.literature_documents.item(index.row(), 0) is not None
        )
        if not document_ids:
            QMessageBox.information(self, "Knowledge Pack", "Select the exact document rows to include.")
            return
        library = self._literature_store.get_library(library_id)
        name, accepted = QInputDialog.getText(self, "Knowledge Pack", "Pack name:", text=library.name)
        if not accepted or not name.strip():
            return
        version, accepted = QInputDialog.getText(self, "Knowledge Pack", "Immutable pack version (SemVer):", text="1.0.0")
        if not accepted or not version.strip():
            return
        pack_id_text, accepted = QInputDialog.getText(
            self,
            "Knowledge Pack identity",
            "Stable Pack ID (reuse this ID when exporting a later version):",
            text=str(uuid4()),
        )
        if not accepted:
            return
        try:
            pack_id = UUID(pack_id_text.strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Knowledge Pack ID", "Enter a valid UUID for the stable pack identity.")
            return
        supplements = ()
        if QMessageBox.question(
            self, "Add researcher-authored note?",
            "Would you like to add a short researcher-authored explanatory note to the pack manifest? It will be included in the pack and may be sensitive.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes:
            note_title, accepted = QInputDialog.getText(self, "Knowledge Pack note", "Note title:")
            if not accepted or not note_title.strip():
                return
            note_text, accepted = QInputDialog.getMultiLineText(
                self, "Knowledge Pack note", "Researcher-authored note content (included in the package manifest):"
            )
            if not accepted or not note_text.strip():
                return
            supplements = (KnowledgePackResource(
                resource_id=uuid4(), kind="researcher_note", title=note_title.strip(),
                metadata=LibraryDocumentMetadata(access_status="unknown", evidence_level="metadata_only"),
                authored_content=note_text,
            ),)
        selected_documents = [self._literature_store.get_document(library_id, item) for item in document_ids]
        selection_preview = "\n".join(
            f"• {item.display_title} · access {item.metadata.access_status} · license {item.metadata.license or 'Unknown'}"
            for item in selected_documents
        )
        if supplements:
            selection_preview += f"\n• researcher note: {supplements[0].title} (included text)"
        if QMessageBox.question(
            self, "Preview Knowledge Pack resources",
            f"Pack: {name.strip()} · {version.strip()} · ID {pack_id}\nSelected resources:\n{selection_preview}\n\n"
            "Unknown licenses remain unknown. Review these resources before export.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        mode = QMessageBox(self)
        mode.setWindowTitle("Choose Knowledge Pack contents")
        mode.setText("Reference-only is the safe default. Content-inclusive export packages retained document files.")
        mode.setInformativeText("Unknown licenses do not grant redistribution rights. Content-inclusive export requires your explicit rights confirmation.")
        mode.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        mode.button(QMessageBox.Yes).setText("Reference-only")
        mode.button(QMessageBox.No).setText("Include retained content…")
        mode.setDefaultButton(QMessageBox.Yes)
        choice = mode.exec()
        if choice == QMessageBox.Cancel:
            return
        include_payloads = choice == QMessageBox.No
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export NeuroScope Knowledge Pack", f"{name.strip()}-{version.strip()}.nskpack", "Knowledge Pack (*.nskpack)"
        )
        if not destination:
            return
        confirmed_redistribution = False
        confirmed_private = False
        if include_payloads:
            if QMessageBox.question(
                self, "Confirm redistribution authorization",
                "Confirm that you are authorized to redistribute every selected document payload in this pack. This confirmation does not change unknown license metadata.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                return
            confirmed_redistribution = True
            if any(item.metadata.access_status != "public" for item in selected_documents):
                if QMessageBox.question(
                    self, "Confirm private or unknown content",
                    "At least one selected document is private or has unknown access status. Confirm separately that including these potentially confidential payloads is authorized.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                ) != QMessageBox.Yes:
                    return
                confirmed_private = True
        try:
            KnowledgePackService(self._literature_store).create_from_library(
                library_id,
                document_ids,
                destination,
                name=name.strip(),
                version=version.strip(),
                pack_id=pack_id,
                supplements=supplements,
                include_document_payloads=include_payloads,
                researcher_confirmed_redistribution=confirmed_redistribution,
                researcher_confirmed_private_payloads=confirmed_private,
            )
            self.literature_results.setPlainText(
                "Knowledge Pack exported as " + ("reference-only; no document contents were included." if not include_payloads else "content-inclusive after explicit authorization confirmation.")
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Knowledge Pack export failed", str(exc))

    def _import_knowledge_pack(self) -> None:
        if self._literature_store is None:
            return
        source, _ = QFileDialog.getOpenFileName(
            self, "Inspect NeuroScope Knowledge Pack", "", "Knowledge Pack (*.nskpack);;ZIP archives (*.zip)"
        )
        if not source:
            return
        service = KnowledgePackService(self._literature_store)
        try:
            preview = service.inspect(source)
            manifest = preview.manifest
            warning_text = "\n\nWarnings:\n• " + "\n• ".join(preview.warnings) if preview.warnings else "\n\nNo package warnings were reported."
            resources = "\n".join(f"• {item.kind}: {item.title} ({item.resource_id})" for item in manifest.resources[:30])
            if len(manifest.resources) > 30:
                resources += f"\n• … and {len(manifest.resources) - 30} more resource(s)"
            prompt = (
                f"{manifest.name} · version {manifest.version}\nPack ID: {manifest.pack_id}\n"
                f"Description: {manifest.description or '—'}\n"
                f"Author: {manifest.author or 'Unknown'} · domains: {', '.join(manifest.scientific_domains) or 'Unknown'}\n"
                f"Compatibility: NeuroScope {manifest.minimum_neuroscope_version} to {manifest.maximum_neuroscope_version or 'no maximum'}\n"
                f"Dependencies: {', '.join(f'{item.pack_id} {item.version}' for item in manifest.dependencies) or 'None'}\n"
                f"Resources: {len(manifest.resources)} · payload files: {preview.payload_count}\n"
                f"License: {manifest.license_declaration or 'Unknown'} · redistribution: {manifest.redistribution_status}\n\n"
                f"{resources}{warning_text}\n\nInspect only, then install inactive? Imported resources remain assistant-hidden until explicitly enabled."
            )
            if QMessageBox.question(self, "Review Knowledge Pack before import", prompt, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            scope_options = ("core", "personal", "laboratory", "project")
            scope, accepted = QInputDialog.getItem(
                self, "Choose local pack scope",
                "Choose an organizational scope for this project-local library. This does not enforce access control.",
                list(scope_options), scope_options.index("project"), False,
            )
            if not accepted:
                return
            installed = service.import_pack(source, activate=False, scope=scope)
            self._refresh_literature_libraries()
            self.literature_results.setPlainText(
                f"Installed {manifest.name} {manifest.version} as inactive library {installed.library_id}. "
                "Review its documents, explicitly mark any assistant-visible, then select the pack library to activate retrieval."
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Knowledge Pack import failed", str(exc))

    def _remove_knowledge_pack(self) -> None:
        library_id = self._current_literature_library_id()
        if self._literature_store is None or not library_id:
            return
        try:
            library = self._literature_store.get_library(library_id)
            if library.knowledge_pack_id is None or library.knowledge_pack_version is None:
                QMessageBox.information(self, "Knowledge Pack", "The selected library is not an imported Knowledge Pack.")
                return
            if QMessageBox.question(
                self, "Remove Knowledge Pack",
                f"Remove {library.name} and its local indexed contents? Other libraries and pack versions will remain unchanged.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                return
            KnowledgePackService(self._literature_store).remove(
                library.knowledge_pack_id, library.knowledge_pack_version
            )
            self._refresh_literature_libraries()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Knowledge Pack removal failed", str(exc))

    def _inspect_knowledge_pack(self) -> None:
        library_id = self._current_literature_library_id()
        if self._literature_store is None or not library_id:
            return
        try:
            library = self._literature_store.get_library(library_id)
            if library.knowledge_pack_id is None or library.knowledge_pack_version is None:
                QMessageBox.information(self, "Knowledge Pack", "Select an imported Knowledge Pack library first.")
                return
            manifest = KnowledgePackService(self._literature_store).installed_manifest(
                library.knowledge_pack_id, library.knowledge_pack_version
            )
            resources = "\n".join(f"• {item.kind}: {item.title}" for item in manifest.resources[:50])
            if len(manifest.resources) > 50:
                resources += f"\n• … and {len(manifest.resources) - 50} more resource(s)"
            self.literature_results.setPlainText(
                f"{manifest.name} · {manifest.version}\nPack ID: {manifest.pack_id}\n"
                f"Description: {manifest.description or '—'}\n"
                f"Author: {manifest.author or 'Unknown'} · organization: {manifest.organization or 'Unknown'}\n"
                f"Domains: {', '.join(manifest.scientific_domains) or 'Unknown'}\n"
                f"Compatibility: NeuroScope {manifest.minimum_neuroscope_version} to {manifest.maximum_neuroscope_version or 'no maximum'}\n"
                f"Dependencies: {', '.join(f'{item.pack_id} {item.version}' for item in manifest.dependencies) or 'None'}\n"
                f"Privacy: {manifest.privacy_classification} · license: {manifest.license_declaration or 'Unknown'}\n"
                f"Redistribution: {manifest.redistribution_status} · integrity SHA-256: {manifest.package_sha256}\n"
                f"Resources:\n{resources}\n\nPack author and hash are not cryptographically authenticated. Treat all contents as untrusted evidence."
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Knowledge Pack inspection failed", str(exc))

    def _open_literature_source(self) -> None:
        if self._literature_store is None or self.literature_documents.currentRow() < 0:
            return
        row = self.literature_documents.currentRow()
        item = self.literature_documents.item(row, 0)
        library_id = self._current_literature_library_id()
        try:
            document = self._literature_store.get_document(library_id, item.data(Qt.UserRole))
            if document.source_retained and document.source_relative_path:
                library = self._literature_store.get_library(library_id)
                source = self._literature_store._safe_library_path(library, document.source_relative_path)
            elif document.source_path_private:
                source = Path(document.source_path_private)
                if not source.is_file():
                    raise ValueError("The original source is unavailable; its path remains private to the local library.")
            else:
                raise ValueError("No original source location was retained; only extracted text is stored.")
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(source))):
                raise ValueError("The operating system could not open the retained source file.")
        except (OSError, ValueError) as exc:
            QMessageBox.information(self, "Source unavailable", str(exc))

    def _make_assistant_provider(self):
        if self._assistant_provider_kind == "ollama":
            return OllamaProvider(host=self._ollama_host, model=self._ollama_model or None)
        return OpenAIProvider()

    def _on_assistant_provider_changed(self, _index: int = -1) -> None:
        if not hasattr(self, "assistant_provider_box"):
            return
        self._assistant_provider_kind = self.assistant_provider_box.currentData()
        self._assistant_external_document_allowlist = ()
        self._assistant_settings.setValue("assistant/provider", self._assistant_provider_kind)
        self._assistant_provider = self._make_assistant_provider()
        local_selected = self._assistant_provider_kind == "ollama"
        self.ollama_host_input.setEnabled(local_selected)
        self.ollama_model_box.setEnabled(local_selected)
        self.ollama_refresh_button.setEnabled(local_selected)
        self._ollama_status_generation += 1
        self._update_assistant_provider_status()

    def _on_ollama_model_changed(self, _index: int = -1) -> None:
        if not hasattr(self, "ollama_model_box"):
            return
        self._ollama_model = str(self.ollama_model_box.currentData() or "")
        self._assistant_settings.setValue("assistant/ollama_model", self._ollama_model)
        if self._assistant_provider_kind == "ollama":
            self._assistant_provider = self._make_assistant_provider()
            self._ollama_status = None
            self._update_assistant_provider_status()

    def _on_ollama_host_changed(self) -> None:
        self._ollama_host = self.ollama_host_input.text().strip() or DEFAULT_OLLAMA_HOST
        try:
            candidate = OllamaProvider(host=self._ollama_host, model=self._ollama_model or None)
        except ValueError as exc:
            self.assistant_activity_label.setText(str(exc))
            return
        self._ollama_host = candidate.host
        self.ollama_host_input.setText(candidate.host)
        self._assistant_settings.setValue("assistant/ollama_host", candidate.host)
        self._ollama_status = None
        if self._assistant_provider_kind == "ollama":
            self._assistant_provider = candidate
            self._update_assistant_provider_status()

    def refresh_ollama_models(self) -> None:
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            self.assistant_activity_label.setText("Wait for the current assistant request before refreshing models.")
            return
        host = self.ollama_host_input.text().strip() or DEFAULT_OLLAMA_HOST
        try:
            self._ollama_host = OllamaProvider(host=host).host
            provider = OllamaProvider(host=self._ollama_host, model=self._ollama_model or None)
        except ValueError as exc:
            self.assistant_activity_label.setText(str(exc))
            return
        if provider.is_remote:
            consent = QMessageBox.question(
                self,
                "Remote Ollama endpoint disclosure",
                f"This checks installed model names and capability metadata at {provider.host}. "
                "The endpoint is not loopback and is treated as a remote service. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if consent != QMessageBox.Yes:
                self.assistant_activity_label.setText("Endpoint check cancelled; no request was sent.")
                return
        if self._ollama_status_thread is not None and self._ollama_status_thread.isRunning():
            self.assistant_activity_label.setText("Ollama status check is already running.")
            return
        self._ollama_status_generation += 1
        generation = self._ollama_status_generation
        self.assistant_activity_label.setText("Checking Ollama service and installed model capabilities…")
        self.ollama_refresh_button.setEnabled(False)
        thread = QThread(self)
        worker = _OllamaStatusWorker(provider, generation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_ollama_status_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_ollama_status_thread_finished)
        self._ollama_status_thread = thread
        self._ollama_status_worker = worker
        thread.start()

    @Slot(int, object, object)
    def _on_ollama_status_finished(self, generation: int, status, error) -> None:
        if generation != self._ollama_status_generation:
            return
        if error is not None:
            self.assistant_activity_label.setText(f"Ollama status check failed: {error}")
            return
        self._ollama_status = status
        wanted = self._ollama_model
        self.ollama_model_box.blockSignals(True)
        self.ollama_model_box.clear()
        for model in status.models:
            self.ollama_model_box.addItem(model, model)
        selected_index = self.ollama_model_box.findData(wanted)
        if selected_index >= 0:
            self.ollama_model_box.setCurrentIndex(selected_index)
        elif status.models:
            self.ollama_model_box.setCurrentIndex(-1)
        self.ollama_model_box.blockSignals(False)
        self.assistant_activity_label.setText(status.message)
        self._update_assistant_provider_status()

    @Slot()
    def _on_ollama_status_thread_finished(self) -> None:
        thread = self._ollama_status_thread
        self._ollama_status_thread = None
        self._ollama_status_worker = None
        self.ollama_refresh_button.setEnabled(self._assistant_provider_kind == "ollama")
        if thread is not None:
            thread.deleteLater()

    @Slot(int)
    def _on_assistant_profile_changed(self, _index: int = -1) -> None:
        if not hasattr(self, "assistant_profile_box"):
            return
        profile_id = self.assistant_profile_box.currentData()
        if not isinstance(profile_id, str):
            return
        self._assistant_scientific_profile = self._scientific_profiles.get(profile_id)
        self._assistant_settings.setValue("assistant/scientific_profile", profile_id)
        self._update_assistant_profile_status()

    def _update_assistant_profile_status(self) -> None:
        if hasattr(self, "assistant_profile_details"):
            profile = self._assistant_scientific_profile
            self.assistant_profile_details.setText(
                f"{profile.description} Profile v{profile.version}. Specialization changes guidance, not model weights."
            )

    def _update_assistant_provider_status(self) -> None:
        if not hasattr(self, "assistant_status_label"):
            return
        if self._assistant_provider_kind == "ollama":
            status = self._ollama_status
            if status is not None and status.selected_model in {None, self._ollama_model}:
                self.assistant_status_label.setText(f"Ollama — {status.state}: {status.message}")
            elif not self._ollama_model:
                self.assistant_status_label.setText("Ollama selected — refresh and select an installed tool-capable model.")
            else:
                self.assistant_status_label.setText(
                    f"Ollama selected ({self._ollama_model}) — service not checked; refresh local models."
                )
            return
        if self._assistant_provider.is_configured:
            self.assistant_status_label.setText(
                f"OpenAI configured ({getattr(self._assistant_provider, 'model', 'configured model')}) "
                "— connection checked when sending"
            )
        else:
            self.assistant_status_label.setText(
                "Offline — set OPENAI_API_KEY and NEUROSCOPE_OPENAI_MODEL to enable chat"
            )

    def _request_document_sharing_consent(self) -> tuple[str, ...] | None:
        """Obtain fresh document-specific permission for this external-provider turn."""
        literature_store = self._effective_literature_store()
        if literature_store is None:
            return ()
        rows = []
        for library_id in literature_store.active_library_ids():
            library = literature_store.get_library(library_id)
            for document in literature_store.list_documents(library_id):
                if document.assistant_visible and document.extraction_status == "indexed":
                    rows.append((library, document))
        if not rows:
            return ()
        dialog = QDialog(self)
        dialog.setWindowTitle("Consent for literature passage sharing")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "The selected external provider may receive exact retrieved passages only from individually checked documents below. "
            "No full document or source file is sent. Consent is limited to this request and current provider. "
            "Private or unknown access status is not treated as authorization."
        ))
        boxes = []
        for library, document in rows:
            access = document.metadata.access_status
            status = document.metadata.publication_status
            box = QCheckBox(
                f"{document.display_title} · {library.name} · {document.document_id} [{access}; {status}]"
            )
            box.setChecked(False)
            boxes.append((document, box))
            layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return tuple(str(document.document_id) for document, box in boxes if box.isChecked())

    def send_chat(self, question: str | None = None, *, retry: bool = False) -> None:
        """Request provider assistance only after per-request context consent."""
        question = self.chat_input.text().strip() if question is None else question.strip()
        if not question:
            return
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            self.assistant_activity_label.setText("A request is already in progress.")
            return
        if not self._assistant_provider.is_configured:
            self._update_assistant_provider_status()
            self.assistant_activity_label.setText("Provider unavailable; no message was sent.")
            self.chat_history.append(
                "<i>Selected provider is unavailable or no model is selected. No provider fallback occurred. "
                "Dataset loading and scientific operations remain local and available.</i>"
            )
            return
        is_ollama = isinstance(self._assistant_provider, OllamaProvider)
        is_external = not is_ollama or self._assistant_provider.is_remote
        self._assistant_external_document_allowlist = ()
        if is_external:
            sharing = self._request_document_sharing_consent()
            if sharing is None:
                self.assistant_activity_label.setText("Request not sent; document-sharing consent was not granted.")
                return
            self._assistant_external_document_allowlist = sharing
        if is_ollama and self._assistant_provider.is_remote:
            consent_title = "Remote Ollama context consent"
            provider_privacy = (
                f"The selected non-loopback Ollama endpoint ({self._assistant_provider.host}) will receive this context. "
                "It is treated as a remote service; review its operator's privacy and retention terms."
            )
        elif is_ollama:
            consent_title = "Local Ollama context notice"
            provider_privacy = (
                f"The selected loopback Ollama service ({self._assistant_provider.host}) processes this context locally. "
                "No OpenAI or other external provider will be used."
            )
        else:
            consent_title = "External assistant context consent"
            provider_privacy = (
                "The OpenAI Responses request disables application-state storage, but provider abuse-monitoring retention "
                "may apply under your account's current data controls. Model usage may incur API charges."
            )
        consent = QMessageBox.question(
            self,
            consent_title,
            (
                "Before sending this request, review the information categories the selected provider can receive:\n\n"
                "• Your chat messages (common local paths and credential-shaped text are redacted).\n"
                "• Registered operation names, descriptions and schemas.\n"
                "• Dataset metadata and compact execution summaries only if a read-only tool requests them.\n"
                "• The current dataset ID and selected channel/sample interval if needed to prepare a typed proposal.\n"
                "• Tool calls and their compact local results may also be included in a later provider turn.\n\n"
                "• Exact passages may be shared only from the individually checked documents in the preceding consent dialog.\n"
                "Raw signal samples, project files, local filesystem paths, and API credentials are not sent. "
                f"{provider_privacy} Tool outputs are untrusted data. Consent is requested again on each send. Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if consent != QMessageBox.Yes:
            self._assistant_external_document_allowlist = ()
            self.assistant_activity_label.setText("Request not sent; consent was not granted.")
            return

        if not retry:
            self.chat_history.append(f"<b>You:</b> {html.escape(question)}")
            self._assistant_conversation.append(ChatMessage(role="user", content=question))
            self._assistant_last_question = question
        self.chat_input.clear()
        self.assistant_activity_label.setText(
            "Sending consented chat context; waiting for provider/tool response…"
        )
        self.assistant_send_button.setEnabled(False)
        self.assistant_provider_box.setEnabled(False)
        self.assistant_profile_box.setEnabled(False)
        self.ollama_host_input.setEnabled(False)
        self.ollama_model_box.setEnabled(False)
        self.ollama_refresh_button.setEnabled(False)
        self.assistant_cancel_button.setEnabled(True)
        self.assistant_retry_button.setEnabled(False)
        cancel_event = Event()
        orchestrator = AssistantOrchestrator(
            self._assistant_provider,
            self._assistant_tools,
            scientific_profile=self._assistant_scientific_profile,
        )
        thread = QThread(self)
        worker = _AssistantWorker(
            orchestrator,
            tuple(self._assistant_conversation),
            self._assistant_generation,
            cancel_event,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_assistant_finished, Qt.QueuedConnection)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_assistant_thread_finished)
        self._assistant_thread = thread
        self._assistant_worker = worker
        self._assistant_cancel_event = cancel_event
        thread.start()

    @Slot(int, object, object)
    def _on_assistant_finished(self, generation: int, turn, error) -> None:
        if generation != self._assistant_generation:
            if generation in self._assistant_cancelled_generations:
                self._assistant_cancelled_generations.discard(generation)
                self.assistant_activity_label.setText(
                    "Request cancelled; late assistant response discarded."
                )
            else:
                self.assistant_activity_label.setText(
                    "Application context changed; late assistant response discarded."
                )
            return
        if error is not None:
            message = scrub_sensitive_text(str(error), limit=1000)
            cancelled = "cancel" in message.lower()
            self.assistant_activity_label.setText(
                "Request cancelled; no late response was applied."
                if cancelled
                else f"Request failed: {message}"
            )
            self.chat_history.append(
                f"<i>{'Assistant request cancelled' if cancelled else 'Assistant request failed'}: "
                f"{html.escape(message)}</i>"
            )
            self.assistant_retry_button.setEnabled(not cancelled)
            return
        library_evidence_used = any(
            result.name == "search_local_scientific_library"
            and isinstance(result.data.get("passages"), list)
            and bool(result.data.get("passages"))
            for result in turn.tool_results
        )
        if library_evidence_used:
            # Do not silently resend passage text or answers that may quote it in
            # any later provider turn after the per-request consent expires.
            self._assistant_conversation = [
                message for message in turn.conversation if message.role == "user"
            ]
        else:
            self._assistant_conversation = list(turn.conversation)
        for result in turn.tool_results:
            prefix = "Read-only tool" if result.ok else "Tool request rejected"
            self.chat_history.append(
                f"<i>{html.escape(prefix)} — {html.escape(result.name)}: "
                f"{html.escape(result.summary)}</i>"
            )
            if result.name == "search_local_scientific_library":
                exact_passages = result.data.get("passages", [])
                if isinstance(exact_passages, list):
                    for source in exact_passages:
                        if isinstance(source, dict):
                            location = source.get("page_number") or source.get("character_range")
                            self.literature_results.append(
                                f"<b>{html.escape(str(source.get('title', 'Unknown title')))}</b> · "
                                f"{html.escape(str(source.get('document_id', '')))} / "
                                f"{html.escape(str(source.get('passage_id', '')))} · "
                                f"{html.escape(str(location))}<br>"
                                f"{html.escape(str(source.get('status_warning') or source.get('publication_status', 'unknown')))}<br>"
                                f"<pre>{html.escape(str(source.get('text', '')))}</pre>"
                            )
        self.chat_history.append(f"<b>Assistant:</b> {html.escape(turn.answer)}")
        self._add_assistant_proposals(turn.proposals)
        if turn.proposals:
            self.assistant_activity_label.setText(
                "Prepared proposal(s); waiting for researcher review and explicit approval."
            )
        elif turn.tool_results:
            self.assistant_activity_label.setText("Answered using local read-only scientific tools.")
        else:
            self.assistant_activity_label.setText("Answered from conversation context.")
        if turn.total_tokens is not None:
            self.assistant_activity_label.setText(
                f"{self.assistant_activity_label.text()} · {turn.total_tokens} tokens"
            )
        self.assistant_retry_button.setEnabled(False)

    @Slot()
    def _on_assistant_thread_finished(self) -> None:
        thread = self._assistant_thread
        self._assistant_thread = None
        self._assistant_worker = None
        self._assistant_cancel_event = None
        self._assistant_external_document_allowlist = ()
        if thread is not None:
            thread.deleteLater()
        self.assistant_send_button.setEnabled(True)
        self.assistant_provider_box.setEnabled(True)
        self.assistant_profile_box.setEnabled(True)
        self.ollama_host_input.setEnabled(self._assistant_provider_kind == "ollama")
        self.ollama_model_box.setEnabled(self._assistant_provider_kind == "ollama")
        self.ollama_refresh_button.setEnabled(self._assistant_provider_kind == "ollama")
        self.assistant_cancel_button.setEnabled(False)

    def cancel_assistant_request(self) -> None:
        if self._assistant_cancel_event is not None:
            self._assistant_cancel_event.set()
            self._assistant_cancelled_generations.add(self._assistant_generation)
            self._assistant_generation += 1
            self.assistant_activity_label.setText(
                "Cancellation requested; late results will be discarded. The remote request may continue until timeout."
            )

    def retry_assistant_request(self) -> None:
        if self._assistant_last_question:
            self.send_chat(self._assistant_last_question, retry=True)

    def clear_assistant_conversation(self) -> None:
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            self.cancel_assistant_request()
            return
        self._assistant_conversation.clear()
        self._assistant_external_document_allowlist = ()
        self._assistant_last_question = None
        self.chat_history.clear()
        self.assistant_activity_label.setText(
            "Local conversation cleared; scientific execution history is unchanged."
        )

    def _add_assistant_proposals(self, proposals) -> None:
        for proposal in proposals:
            if not isinstance(proposal, AssistantProposal):
                continue
            item = QListWidgetItem(
                f"{proposal.operation_id} v{proposal.operation_version} — prepared only"
            )
            item.setData(Qt.UserRole, proposal.proposal_id)
            self.assistant_proposal_list.addItem(item)
        if proposals and self.assistant_proposal_list.count():
            self.assistant_proposal_list.setCurrentRow(
                self.assistant_proposal_list.count() - len(proposals)
            )

    def _on_assistant_proposal_selected(self, current, _previous=None) -> None:
        proposal_id = current.data(Qt.UserRole) if current is not None else None
        if not proposal_id:
            self.assistant_proposal_details.clear()
            self.assistant_approve_button.setEnabled(False)
            self.assistant_dismiss_button.setEnabled(False)
            return
        try:
            proposal, prepared = self._assistant_tools.pending_proposal(proposal_id)
        except (KeyError, ValueError) as exc:
            self.assistant_proposal_details.setPlainText(
                f"Stale or unavailable proposal: {exc}\nIt cannot be approved."
            )
            self.assistant_approve_button.setEnabled(False)
            self.assistant_dismiss_button.setEnabled(True)
            return
        lines = [
            "Prepared only — not approved or executed",
            f"Operation: {proposal.operation_id} v{proposal.operation_version}",
            f"Implementation: {proposal.implementation_version}",
            "Exact local input: "
            + json.dumps(prepared.input_data.model_dump(mode="json"), ensure_ascii=False),
            f"Parameters: {json.dumps(proposal.parameter_snapshot, ensure_ascii=False)}",
            "Metadata: "
            + json.dumps(
                [item.model_dump(mode="json") for item in prepared.context.metadata],
                ensure_ascii=False,
            ),
            "Warnings: " + ("; ".join(proposal.warnings) or "none"),
            "Limitations: " + ("; ".join(proposal.limitations) or "none"),
        ]
        self.assistant_proposal_details.setPlainText("\n".join(lines))
        self.assistant_approve_button.setEnabled(self.store is not None)
        self.assistant_dismiss_button.setEnabled(True)

    def _approve_assistant_proposal(self) -> None:
        item = self.assistant_proposal_list.currentItem()
        proposal_id = item.data(Qt.UserRole) if item is not None else None
        if not proposal_id:
            return
        try:
            proposal, prepared = self._assistant_tools.pending_proposal(proposal_id)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Proposal is stale", str(exc))
            self._on_assistant_proposal_selected(item)
            return
        self._assistant_approval_proposal_id = proposal_id
        self._prepared_execution = prepared
        self._prepared_mua_batch = None
        self.operation_review.setPlainText(
            "Assistant-prepared proposal. Review the exact typed request below.\n\n"
            f"Operation: {proposal.operation_id} v{proposal.operation_version}\n"
            f"Implementation: {proposal.implementation_version}\n"
            f"Input: {json.dumps(prepared.input_data.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Parameters: {json.dumps(proposal.parameter_snapshot, ensure_ascii=False)}\n"
            "Warnings: " + "; ".join(proposal.warnings) + "\n"
            "Limitations: " + "; ".join(proposal.limitations) + "\n"
            "This proposal has not executed. Approval below is a separate researcher action."
        )
        self.approve_and_run()

    def _dismiss_assistant_proposal(self) -> None:
        item = self.assistant_proposal_list.currentItem()
        if item is None:
            return
        proposal_id = item.data(Qt.UserRole)
        self._assistant_tools.dismiss_proposal(proposal_id)
        if self._assistant_approval_proposal_id == proposal_id:
            self._assistant_approval_proposal_id = None
            self._prepared_execution = None
        row = self.assistant_proposal_list.row(item)
        self.assistant_proposal_list.takeItem(row)
        self.assistant_proposal_details.clear()
        self.assistant_activity_label.setText("Proposal dismissed; no scientific operation was executed.")

    def closeEvent(self, event):
        """Keep session-owned sources alive until worker completion; then close them."""
        if self._execution_in_flight or (
            self._worker_thread is not None and self._worker_thread.isRunning()
        ) or any(thread.isRunning() for thread, _worker in self._view_workers) or (
            self._assistant_thread is not None and self._assistant_thread.isRunning()
        ) or (
            self._ollama_status_thread is not None and self._ollama_status_thread.isRunning()
        ):
            if self._assistant_thread is not None and self._assistant_thread.isRunning():
                self.cancel_assistant_request()
            QMessageBox.warning(
                self,
                "Analysis running",
                "Wait for active analysis, artifact reads, assistant requests, or Ollama model checks to finish before closing NeuroScope AI.",
            )
            event.ignore()
            return
        if self.active_registration is not None:
            try:
                self.dataset_store.unregister(self.active_registration.session_handle)
            except DatasetAccessError:
                if self.active_dataset is not None:
                    self.active_dataset.close()
        elif self.active_dataset is not None:
            self.active_dataset.close()
        self.active_dataset = None
        self.active_registration = None
        super().closeEvent(event)

