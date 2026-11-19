# NeuroScope AI

NeuroScope AI is an early desktop scaffold for local neural data inspection and deterministic negative threshold spike detection. The LLM, literature retrieval, MUA and PSTH workflows are not implemented.

## Supported input

- `.npy`: numeric NumPy arrays; object arrays are disabled.
- `.csv`: numeric comma-delimited data by default. `delimiter`, `header` and `skiprows` can be supplied to `load_array`.
- `.txt`: numeric text; delimiter must be supplied explicitly. `header` and `skiprows` are also available.
- `.mat`: numeric variables in classic MATLAB files and top-level numeric datasets in v7.3/HDF5 files. If there is more than one candidate, pass `variable_name`. v7.3 nested/group structures are not traversed and MATLAB dimension conventions are not changed automatically.

All formats reject empty, non-numeric, non-finite and non-1D/2D arrays. For 2D arrays, the UI asks whether rows or columns are channels before creating and saving the dataset profile. One-dimensional arrays represent a single signal. Sampling rate, units and signal kind remain unknown until a future explicit metadata editing workflow; analysis parameters do not confirm those fields.

## Install and run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
neuroscope
```

## Current architecture

`src/neuroscope/` contains data loading, domain/profile models, deterministic spike analysis, local JSON persistence and a PySide6 interface. The optional assistant client is not connected to an analysis workflow.

Run focused tests with `pytest`.
