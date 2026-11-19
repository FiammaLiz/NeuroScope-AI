from __future__ import annotations
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QComboBox, QTextEdit,
    QSplitter, QMessageBox, QInputDialog)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from ..data.loaders import load_array, DataLoadError
from ..domain import DatasetProfile, DatasetMetadata
from ..analysis.spikes import detect_negative_spikes
from ..memory.project_store import ProjectStore


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScope AI — Neural Data Analysis")
        self.resize(1280, 800)
        self.current_array: np.ndarray | None = None
        self.profile: DatasetProfile | None = None
        self.store: ProjectStore | None = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        self.load_button = QPushButton("Load dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.project_label = QLabel("No dataset loaded")
        top.addWidget(self.load_button)
        top.addWidget(self.project_label, 1)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Dataset profile"))
        self.profile_view = QTextEdit(); self.profile_view.setReadOnly(True)
        left_layout.addWidget(self.profile_view)
        left_layout.addWidget(QLabel("Analysis settings"))
        self.channel_box = QComboBox(); self.channel_box.addItem("Select channel")
        self.sampling_rate = QLineEdit(); self.sampling_rate.setPlaceholderText("Sampling rate (Hz) — required")
        self.threshold = QLineEdit("-4")
        self.run_button = QPushButton("Detect spikes")
        self.run_button.clicked.connect(self.run_spike_detection)
        left_layout.addWidget(QLabel("Channel")); left_layout.addWidget(self.channel_box)
        left_layout.addWidget(QLabel("Sampling rate (Hz)")); left_layout.addWidget(self.sampling_rate)
        left_layout.addWidget(QLabel("Threshold (noise SD; negative)")); left_layout.addWidget(self.threshold)
        left_layout.addWidget(self.run_button)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.figure = Figure(figsize=(7, 4)); self.canvas = FigureCanvasQTAgg(self.figure)
        right_layout.addWidget(self.canvas, 3)
        right_layout.addWidget(QLabel("Research assistant"))
        self.chat_history = QTextEdit(); self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit(); self.chat_input.setPlaceholderText("Ask about this dataset or its analysis…")
        self.chat_input.returnPressed.connect(self.send_chat)
        right_layout.addWidget(self.chat_history, 1); right_layout.addWidget(self.chat_input)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([340, 940])
        layout.addWidget(splitter)
        self.setCentralWidget(root)

    def load_dataset(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open neural dataset", "",
            "Supported data (*.npy *.mat *.csv *.txt)")
        if not filename: return
        path = Path(filename)
        delimiter = None
        if path.suffix.lower() == ".txt":
            delimiter, ok = QInputDialog.getText(self, "Text delimiter", "Enter delimiter (\\t for tab):", text="\\t")
            if not ok: return
            delimiter = "\t" if delimiter == "\\t" else delimiter
        try:
            array = load_array(path, delimiter=delimiter)
        except DataLoadError as exc:
            QMessageBox.critical(self, "Could not load dataset", str(exc)); return
        self.current_array = np.asarray(array)
        self.channel_box.clear()
        channel_axis = None
        if self.current_array.ndim == 1:
            self.channel_box.addItem("1", 0)
        elif self.current_array.ndim == 2:
            # Orientation is deliberately user-selected, not inferred from shape.
            axis, ok = QInputDialog.getItem(self, "Array orientation",
                "Which axis represents channels?", ["Axis 0 (rows are channels)", "Axis 1 (columns are channels)"], 0, False)
            if not ok: return
            self.channel_axis = 0 if axis.startswith("Axis 0") else 1
            channel_axis = self.channel_axis
            n_channels = self.current_array.shape[self.channel_axis]
            for i in range(n_channels): self.channel_box.addItem(f"Channel {i + 1}", i)
        else:
            QMessageBox.warning(self, "Unsupported dimensions", "The MVP currently supports 1D or 2D arrays."); return
        self.profile = DatasetProfile.from_array(path, self.current_array, channel_axis=channel_axis)
        self.store = ProjectStore(path.parent / f".{path.stem}_neuroscope")
        self.store.save_profile(self.profile)
        self.project_label.setText(path.name)
        self.profile_view.setPlainText(self.profile.model_dump_json(indent=2))
        self._plot_preview()

    def _selected_signal(self):
        if self.current_array is None: raise ValueError("Load a dataset first.")
        idx = self.channel_box.currentData()
        if idx is None: idx = 0
        if self.current_array.ndim == 1: return self.current_array
        axis = getattr(self, "channel_axis", 0)
        return self.current_array[idx, :] if axis == 0 else self.current_array[:, idx]

    def _plot_preview(self):
        signal = self._selected_signal()
        self.figure.clear(); ax = self.figure.add_subplot(111)
        n = min(signal.size, 10000)
        ax.plot(np.arange(n), signal[:n], linewidth=0.65)
        ax.set_title("Signal preview (sample index; sampling rate not yet confirmed)")
        ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude")
        self.canvas.draw_idle()

    def run_spike_detection(self):
        try:
            signal = self._selected_signal()
            fs = float(self.sampling_rate.text())
            threshold_sd = float(self.threshold.text())
            result = detect_negative_spikes(signal, fs, threshold_sd)
        except Exception as exc:
            QMessageBox.warning(self, "Analysis not run", str(exc)); return
        self.figure.clear(); ax = self.figure.add_subplot(111)
        t = np.arange(signal.size) / fs
        ax.plot(t, signal, linewidth=0.55, label="Signal")
        ax.axhline(result["threshold"], color="red", linewidth=1, label="Threshold")
        ax.scatter(result["times_s"], result["amplitudes"], s=10, label="Detected events", zorder=3)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend()
        ax.set_title(f"Negative threshold detection — {len(result['times_s'])} events")
        self.canvas.draw_idle()
        if self.profile:
            # Analysis parameters are not confirmed dataset metadata.
            if self.store:
                self.store.append_analysis("negative_spike_detection",
                    {"sampling_rate_hz": fs, "threshold_sd": threshold_sd,
                     "channel": self.channel_box.currentText()},
                    f"Detected {len(result['times_s'])} negative-going events.")
        self.chat_history.append(f"<b>Analysis completed:</b> detected {len(result['times_s'])} events. Parameters: fs={fs:g} Hz, threshold={threshold_sd:g} noise SD.")

    def send_chat(self):
        question = self.chat_input.text().strip()
        if not question: return
        self.chat_history.append(f"<b>You:</b> {question}")
        self.chat_history.append("<i>LLM integration is the next milestone. Dataset profile and analysis history are already stored locally.</i>")
        self.chat_input.clear()
